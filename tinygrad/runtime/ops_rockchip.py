# pylint: disable=cell-var-from-loop
# a python uops emulator
# works to test the tensor cores, and all the uops in general
# this is the (living) definition of uops
import array
import ctypes
import functools
import mmap
import os
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
import pickle, base64, itertools, time, struct, sys
from tinygrad.dtype import DType, dtypes, ImageDType, PtrDType, truncate
from tinygrad.helpers import all_same, getenv, flatten, get_single_element, mv_address, to_mv
from tinygrad.device import BufferSpec, Compiled, Compiler, Allocator
from tinygrad.codegen.opt import tc
from tinygrad.runtime.ops_cpu import HCQBuffer
from tinygrad.runtime.support.hcq import FileIOInterface, HCQAllocatorBase
from tinygrad.uop.ops import exec_alu, Ops, UOp, GroupOp, PatternMatcher, UPat
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen import rockchip as rk

import sys, numpy as np
np.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=False)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

DEBUG = getenv("DEBUG")
FUSE_POSTOPS = getenv("ROCKCHIP_FUSE_POSTOPS", 1)

@dataclass(frozen=True)
class RockchipConvInfo:
  metadata: tuple[str, ...]
  axes: tuple[int, ...]
  lhs_shape: tuple[Any, ...]
  rhs_shape: tuple[Any, ...]
  out_shape: tuple[Any, ...]
  lhs_base_shape: tuple[int, ...]
  rhs_base_shape: tuple[int, ...]
  out_base_shape: tuple[int, ...]
  lhs_tensor_shape: tuple[int, ...]
  rhs_tensor_shape: tuple[int, ...]
  out_tensor_shape: tuple[int, ...]
  post_ops: tuple[tuple[Ops, Any], ...] = ()

def _rockchip_conv_rewrite(red:UOp) -> UOp|None:
  if len(red.src) != 1: return None
  def _peel_mul(u:UOp) -> UOp|None:
    visited:set[UOp] = set()
    while True:
      if u in visited: return None
      visited.add(u)
      if u.op is Ops.MUL: return u
      if u.op in {Ops.CAST, Ops.RESHAPE, Ops.PERMUTE, Ops.SHRINK, Ops.EXPAND, Ops.VIEW, Ops.CONTIGUOUS} and len(u.src) == 1:
        u = u.src[0]
        continue
      return None

  mul = _peel_mul(red.src[0])
  if mul is None: return None
  lhs, rhs = mul.src
  def _metadata_names(uop:UOp) -> tuple[str, ...]:
    return tuple(m.name for m in (uop.metadata or ())) if hasattr(uop, "metadata") else tuple()
  meta_names = _metadata_names(red)
  if DEBUG >= 3:
    print("ROCKCHIP rewrite candidate", meta_names, red.arg, tuple(red.shape),
          lhs.op, tuple(getattr(lhs, "shape", ())), tuple(getattr(lhs, "full_shape", ())), tuple(getattr(lhs.base, "shape", ())) if hasattr(lhs, "base") else (),
          rhs.op, tuple(getattr(rhs, "shape", ())), tuple(getattr(rhs, "full_shape", ())), tuple(getattr(rhs.base, "shape", ())) if hasattr(rhs, "base") else ())
  if not any(name.startswith("conv") for name in meta_names):
    axes = tuple(red.arg[1]) if isinstance(red.arg, tuple) and len(red.arg) == 2 else tuple()
    if not axes or sorted(axes) != list(axes):
      if DEBUG >= 3: print("ROCKCHIP rewrite reject axes", axes)
      return None
    if any(a < len(red.shape)-len(axes) for a in axes):
      if DEBUG >= 3: print("ROCKCHIP rewrite reject axis position", axes, len(red.shape))
      return None
    if lhs.shape != rhs.shape:
      if DEBUG >= 3: print("ROCKCHIP rewrite reject shape mismatch", lhs.shape, rhs.shape)
      return None
  axes = tuple(red.arg[1]) if isinstance(red.arg, tuple) and len(red.arg) == 2 else tuple()
  movement_ops = {Ops.RESHAPE, Ops.PERMUTE, Ops.SHRINK, Ops.EXPAND, Ops.VIEW, Ops.CONTIGUOUS, Ops.CAST, Ops.BITCAST}
  terminal_ops = {Ops.STORE, Ops.SINK, Ops.ASSIGN, Ops.KERNEL}
  def _downstream_ok(node:UOp, seen:set[UOp]) -> bool:
    for ref in list(node.children):
      child = ref()
      if child is None or child in seen: continue
      seen.add(child)
      if child.op in terminal_ops: continue
      if child.op in movement_ops:
        if not _downstream_ok(child, seen):
          return False
        continue
      if DEBUG >= 3:
        print("ROCKCHIP rewrite skip due to downstream op", child.op)
      return False
    return True
  if not _downstream_ok(red, set()):
    return None
  def _to_int(x):
    if isinstance(x, int): return x
    if hasattr(x, "__int__"): return int(x)
    if hasattr(x, "vmax") and hasattr(x, "vmin") and x.vmax == x.vmin:
      return int(x.vmax)
    raise ValueError(f"unable to convert shape element {x}")
  def _safe_shape(u:UOp) -> tuple[int, ...]:
    shape = getattr(u, "shape", ())
    return tuple(_to_int(x) for x in shape)
  def _base_shape(u:UOp) -> tuple[int, ...]:
    base = getattr(u, "base", None)
    if base is None: return tuple()
    bshape = getattr(base, "shape", ())
    if not bshape:
      ptr = getattr(base, "ptrdtype", None)
      if ptr is not None:
        sz = getattr(ptr, "size", 0)
        if sz: return (sz,)
    return tuple(_to_int(x) for x in bshape) if bshape else tuple()
  def _tensor_shape(u:UOp, base_shape:tuple[int, ...]) -> tuple[int, ...]:
    try:
      target = int(np.prod(base_shape)) if base_shape else int(np.prod(_safe_shape(u)))
    except Exception:
      target = int(np.prod(_safe_shape(u)))
    best: tuple[int, ...]|None = None
    best_score = -float('inf')
    for node in u.toposort():
      if node.op is Ops.RESHAPE and isinstance(node.arg, tuple):
        try:
          candidate = tuple(_to_int(x) for x in node.arg)
          if int(np.prod(candidate)) == target:
            score = -abs(len(candidate)-4)
            if len(candidate) == 1: score -= 1  # avoid scalar shapes when possible
            if score > best_score:
              best_score = score
              best = candidate
        except Exception:
          continue
    if best is not None: return best
    if base_shape: return base_shape
    return _safe_shape(u)
  movement_ops = {Ops.RESHAPE, Ops.PERMUTE, Ops.SHRINK, Ops.EXPAND, Ops.VIEW, Ops.CONTIGUOUS, Ops.CAST, Ops.BITCAST}
  terminal_ops = {Ops.STORE, Ops.SINK, Ops.ASSIGN, Ops.KERNEL}
  fusible_ops = {Ops.ADD} if FUSE_POSTOPS else set()

  def _extract_const(node:UOp) -> Any|None:
    cur = node
    seen:set[UOp] = set()
    while True:
      if cur in seen: return None
      seen.add(cur)
      if cur.op is Ops.CONST: return cur.arg
      if cur.op in movement_ops and len(cur.src) == 1:
        cur = cur.src[0]
        continue
      return None

  def _downstream_info(root:UOp) -> tuple[bool, list[tuple[Ops, Any]]]:
    post:list[tuple[Ops, Any]] = []
    seen:set[UOp] = set()
    stack:list[UOp] = [root]
    while stack:
      node = stack.pop()
      for ref in list(node.children):
        child = ref()
        if child is None or child in seen: continue
        seen.add(child)
        if DEBUG >= 3:
          print("ROCKCHIP downstream inspect", node.op, "->", child.op)
        if child.op in movement_ops:
          stack.append(child)
          continue
        if child.op in fusible_ops:
          const_operand:UOp|None = None
          conv_operand:UOp|None = None
          for src in child.src:
            if src is root or root in src.parents:
              conv_operand = src
            else:
              const_operand = src
          if FUSE_POSTOPS and conv_operand is not None and const_operand is not None:
            const_val = _extract_const(const_operand)
            if const_val is None:
              return False, []
            post.append((child.op, const_val))
            stack.append(child)
            continue
          return False, []
        if child.op in terminal_ops:
          continue
        return False, []
    return True, post

  ok, post_ops = _downstream_info(red)
  if not ok:
    return None

  info = RockchipConvInfo(meta_names, axes,
    _safe_shape(lhs), _safe_shape(rhs), _safe_shape(red),
    (_bs_lhs:=_base_shape(lhs)),
    (_bs_rhs:=_base_shape(rhs)),
    (_bs_out:=_base_shape(red)),
    _tensor_shape(lhs, _bs_lhs),
    _tensor_shape(rhs, _bs_rhs),
    _tensor_shape(red, _bs_out),
    tuple(post_ops))
  if DEBUG >= 2:
    parent_ops = [p.op for p in red.parents]
    print("ROCKCHIP conv rewrite applied", info, "parents", parent_ops)
  custom = UOp(Ops.CUSTOM, red.dtype, src=mul.src, arg=info, metadata=red.metadata)
  return custom

rockchip_conv_pm = PatternMatcher([
  (UPat(Ops.REDUCE_AXIS, name="red"), lambda red: _rockchip_conv_rewrite(red)),
])

def rockchip_conv_prepass(ast:UOp) -> UOp:
  """
  Apply Rockchip-specific conv rewrites on the high level AST before generic lowering.
  """
  @functools.cache
  def _rewrite(u:UOp) -> UOp:
    # rewrite children first
    if len(u.src):
      new_src = tuple(_rewrite(s) for s in u.src)
      if new_src != u.src: u = u.replace(src=new_src)

    if DEBUG >= 3 and u.op is Ops.REDUCE_AXIS:
      print("ROCKCHIP prepass inspect reduce", u.arg, tuple(u.shape), [s.op for s in u.src])
      if DEBUG >= 4:
        from tinygrad.uop.ops import pyrender
        print("\\n".join(pyrender(u)[:15]))

    # attempt local rewrite on this node
    rewritten = rockchip_conv_pm.rewrite(u)
    if rewritten is not None and rewritten is not u:
      return _rewrite(rewritten)
    return u

  return _rewrite(ast)

def storage_fmt_for_dtype(dtype: DType): return 'H' if dtype == dtypes.bfloat16 else dtype.fmt

def to_storage_scalar(x, dtype: DType):
  if dtype == dtypes.bfloat16: return (struct.unpack('I', struct.pack('f', float_to_bf16(x)))[0] >> 16) & 0xFFFF
  return x

def from_storage_scalar(x, dtype: DType):
  if dtype == dtypes.bfloat16: return struct.unpack('f', struct.pack('I', (x & 0xFFFF) << 16))[0]
  return x

def _load(m, i, dtype: DType):
  if i is None: return 0.0
  if i < 0 or i >= len(m): raise IndexError(f"load out of bounds, size is {len(m)} and access is {i}")
  return from_storage_scalar(m[i], dtype)

def load(inp, j, dtype: DType):
  if len(inp) == 2: return [_load(m, x+j if x is not None else None, dtype) if gate else default for (m,x,gate),default in zip(*inp)]
  return [_load(m, x+j if x is not None else None, dtype) for m,x,_ in inp[0]]

def _store(m, i, v, dtype: DType):
  if i < 0 or i >= len(m): raise IndexError(f"store out of bounds, size is {len(m)}, access is {i}, value is {v}")
  m[i] = to_storage_scalar(v, dtype)

class RockchipRenderer(Renderer):
  device = "ROCKCHIP"
  code_for_op = {
    # Ops.MAX: 0, 
    Ops.ADD: 2, 
    # Ops.FDIV: 3, 
    # Ops.IDIV: 3, 
    # Ops.SUB: 4, 
    # Ops.NEG: 6, 
    Ops.MUL: None
    }
  pre_matcher = rockchip_conv_pm

  def preprocess_ast(self, ast:UOp) -> UOp:
    return rockchip_conv_prepass(ast)

  def render(self, uops:list[UOp]) -> str:
    conv = next((u for u in uops if u.op is Ops.CUSTOM and isinstance(u.arg, RockchipConvInfo)), None)
    if conv is not None:
      def _find_global_id(node:UOp) -> int|None:
        for parent in node.toposort():
          if parent.op is Ops.DEFINE_GLOBAL:
            return parent.arg
        return None

      store_uop = next((u for u in uops if u.op is Ops.STORE and (conv in u.src or conv in u.parents)), None)
      if store_uop is None:
        if DEBUG:
          print("RK_CONV render fallback: no direct STORE for conv output")
      else:
        out_gid = _find_global_id(store_uop.src[0])
        lhs_gid = _find_global_id(conv.src[0])
        rhs_gid = _find_global_id(conv.src[1])
        if None in (out_gid, lhs_gid, rhs_gid):
          if DEBUG:
            print("RK_CONV render fallback: missing global ids", out_gid, lhs_gid, rhs_gid)
        else:
          globals_order = tuple(u.arg for u in uops if u.op is Ops.DEFINE_GLOBAL)
          metadata = {"globals_order": globals_order, "out": out_gid, "lhs": lhs_gid, "rhs": rhs_gid}
          payload = ("RK_CONV", conv.dtype, conv.arg, metadata)
          return base64.b64encode(pickle.dumps(payload)).decode()

    # the value of SPECIAL comes from local/global_size, not form its source
    lops = [(u.op, u.dtype, [uops.index(v) for v in u.src if u.op is not Ops.SPECIAL], u.arg) for u in uops]
    return base64.b64encode(pickle.dumps(lops)).decode()


class RockchipDevice(Compiled):
  def create_flink_name(self, handle: int) -> int:
    """
    Create a flink name for a GEM handle using DRM_IOCTL_GEM_FLINK.
    Args:
      handle: The GEM handle to create a flink name for
      
    Returns:
      The flink name (uint32) on success, raises exception on failure
    """
    flink_req = rk.struct_drm_gem_flink(handle=handle, name=0)
    
    try:
      result = rk.DRM_IOCTL_GEM_FLINK(self.fd_ctl, __payload=flink_req)
      
      print(f"SUCCESS: Created flink name {flink_req.name} for handle {handle}")
      return flink_req.name
    except Exception as e:
      print(f"ERROR: DRM_IOCTL_GEM_FLINK failed: {e}")
      raise

  def _gpu_alloc(self, size:int, flags) -> HCQBuffer:
    mem_create = rk.DRM_IOCTL_RKNPU_MEM_CREATE(self.fd_ctl, size=size, flags=flags | rk.RKNPU_MEM_NON_CACHEABLE)
    mem_map = rk.DRM_IOCTL_RKNPU_MEM_MAP(self.fd_ctl, handle=mem_create.handle, offset=0)    
    va_addr = self.fd_ctl.mmap(0, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, mem_map.offset)

    # Create flink name for the GEM handle
    flink_name = self.create_flink_name(mem_create.handle)
    # Store flink name in meta for later use
    mem_create.flink_name = flink_name

    return HCQBuffer(va_addr=va_addr, size=size, meta=mem_create)

  def __init__(self, device:str): 
    self.fd_ctl = FileIOInterface(f"/dev/dri/card1", os.O_RDWR)
    self.cmd_buf = self._gpu_alloc(1024, 0)
    self.task_buf = self._gpu_alloc(1024, rk.RKNPU_MEM_KERNEL_MAPPING)

    self.input_buf = None
    self.weight_buf = None
    self.output_buf = None

    self.buffer_list = []
    self.code_for_op = RockchipRenderer.code_for_op

    super().__init__(device, RockchipAllocator(self), RockchipRenderer(), RockchipCompiler(), functools.partial(RockchipProgram, self))

  def add_buffer(self, size):

    self.input_buf = next((item["buf"] for item in self.buffer_list if item["buf_type"] == "input" and item["size"] == size), None)
    self.weight_buf = next((item["buf"] for item in self.buffer_list if item["buf_type"] == "weight" and item["size"] == size), None)
    self.output_buf = next((item["buf"] for item in self.buffer_list if item["buf_type"] == "output" and item["size"] == size), None)
    if (self.input_buf is None or self.weight_buf is None or self.output_buf is None):
      self.input_buf = self._gpu_alloc(size, 0)
      self.buffer_list.append({"buf_type": "input", "buf": self.input_buf, "size": size})
      self.weight_buf = self._gpu_alloc(size, 0)
      self.buffer_list.append({"buf_type": "weight", "buf": self.weight_buf, "size": size})
      self.output_buf = self._gpu_alloc(size, 0)
      self.buffer_list.append({"buf_type": "output", "buf": self.output_buf, "size": size})

class RockchipProgram:


  def reg(self, val, shift, mask):
    return ((val) << shift) & mask;
  def emit_raw(self, target, reg, value):
    # Pack the values into a 64-bit integer as per hardware spec
    target = target + 0x1
    packed_value = ((target & 0xFFFF) << 48) | ((value & 0xFFFFFFFF) << 16) | (reg & 0xFFFF)

    self.q.append(packed_value)
  
  def get_precision(self, dtype, fp32out=False):
    # 3'd0: Integer 8bit; 
    # 3'd1: Integer 16bit; 
    # 3'd2: Float point 16bit; 
    # 3'd3: Bfloat 16bit; 
    # 3'd4: Integer 32bit; 
    # 3'd5: Float point 32bit; 
    # 3'd6: Integer 4bit. 
    from tinygrad import dtypes
    if dtype == dtypes.int8:
      return 0
    elif dtype == dtypes.int16:
      return 1
    elif dtype == dtypes.float16:
      if fp32out:
        return 5
      else:
        return 2
    elif dtype == dtypes.bfloat16:
      return 3
    elif dtype == dtypes.int32:
      return 4
    elif dtype == dtypes.float32:
      return 5
    elif getattr(dtype, "itemsize", None) == 0.5 or getattr(dtype, "name", "") == "int4":
      return 6
    else:
      raise ValueError(f"Unsupported dtype for precision: {dtype}")
  def get_edata_size(self, dtype):
    # edata_size / erdma_data_size
    # Data size of the cube from ERDMA. 
    # 2'd0: 4bit; -> Not Supported
    # 2'd1: 8bit; 
    # 2'd2: 16bit; 
    # 2'd3: 32bit
    if dtype == dtypes.int8:
      return 1
    elif dtype == dtypes.int16 or dtype == dtypes.float16 or dtype == dtypes.bfloat16:
      return 2
    elif dtype == dtypes.int32 or dtype == dtypes.float32:
      return 3
    else:
      raise ValueError(f"Unsupported dtype for edata_size: {dtype}")
  def get_is_fp16(self, dtype):
    return dtype == dtypes.float16 or dtype == dtypes.float

  def ops(self, op, dtype):
    # print(op, dtype, self.get_precision(dtype), op==Ops.ADD)

    self.emit_raw(rk.DPU, rk.REG_DPU_DATA_FORMAT,
      # self.reg(self.get_precision(dtype, fp32out=op==Ops.ADD), rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
      self.reg(self.get_precision(dtype), rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
      self.reg(self.get_precision(dtype), rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
      self.reg(self.get_precision(dtype), rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))

    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
      self.reg(self.get_precision(dtype), rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__MASK) |
      self.reg(15, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      self.reg(0, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE__MASK) |
      self.reg(self.get_precision(dtype), rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__MASK) |
      self.reg(0, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE__MASK) |
      self.reg(self.get_is_fp16(dtype), rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__MASK) |
      self.reg(0, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE__MASK) |
      self.reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__MASK))

    self.emit_raw(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE, 
      self.reg(self.get_is_fp16(dtype), rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__SHIFT, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__MASK) |
      # self.reg(0, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__SHIFT, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__MASK) |
      self.reg(1, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__SHIFT, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__MASK));

    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
      self.reg(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__MASK) |
      self.reg(self.get_edata_size(dtype), rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__MASK))
    
    self.emit_raw(rk.DPU, rk.REG_DPU_BS_CFG,
      self.reg(0, rk.DPU_BS_CFG_BS_ALU_ALGO__SHIFT, rk.DPU_BS_CFG_BS_ALU_ALGO__MASK) |
      self.reg(0, rk.DPU_BS_CFG_BS_ALU_SRC__SHIFT, rk.DPU_BS_CFG_BS_ALU_SRC__MASK) |
      self.reg(0, rk.DPU_BS_CFG_BS_RELUX_EN__SHIFT, rk.DPU_BS_CFG_BS_RELUX_EN__MASK) |
      self.reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK) |
      self.reg(0, rk.DPU_BS_CFG_BS_MUL_PRELU__SHIFT, rk.DPU_BS_CFG_BS_MUL_PRELU__MASK) |
      self.reg(1, rk.DPU_BS_CFG_BS_MUL_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_MUL_BYPASS__MASK) |
      self.reg(1, rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_ALU_BYPASS__MASK) |
      self.reg(1, rk.DPU_BS_CFG_BS_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_BYPASS__MASK))

    self.emit_raw(rk.DPU, rk.REG_DPU_BN_CFG,
      self.reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK) |
      self.reg(1, rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_MUL_BYPASS__MASK) |
      self.reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK) |
      self.reg(1, rk.DPU_BN_CFG_BN_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_BYPASS__MASK))

    if op == Ops.MUL:
      self.emit_raw(rk.DPU, rk.REG_DPU_EW_CFG,
        self.reg(0, rk.DPU_EW_CFG_EW_CVT_TYPE__SHIFT, rk.DPU_EW_CFG_EW_CVT_TYPE__MASK) |
        self.reg(0, rk.DPU_EW_CFG_EW_CVT_ROUND__SHIFT, rk.DPU_EW_CFG_EW_CVT_ROUND__MASK) |
        self.reg(1, rk.DPU_EW_CFG_EW_DATA_MODE__SHIFT, rk.DPU_EW_CFG_EW_DATA_MODE__MASK) |
        self.reg(self.get_edata_size(dtype), rk.DPU_EW_CFG_EDATA_SIZE__SHIFT, rk.DPU_EW_CFG_EDATA_SIZE__MASK) |
        self.reg(0, rk.DPU_EW_CFG_EW_EQUAL_EN__SHIFT, rk.DPU_EW_CFG_EW_EQUAL_EN__MASK) |
        self.reg(0, rk.DPU_EW_CFG_EW_BINARY_EN__SHIFT, rk.DPU_EW_CFG_EW_BINARY_EN__MASK) |
        self.reg(0, rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT, rk.DPU_EW_CFG_EW_ALU_ALGO__MASK) |
        self.reg(0, rk.DPU_EW_CFG_EW_RELUX_EN__SHIFT, rk.DPU_EW_CFG_EW_RELUX_EN__MASK) |
        self.reg(1, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
        self.reg(1, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK) |
        self.reg(1, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
        self.reg(1, rk.DPU_EW_CFG_EW_OP_SRC__SHIFT, rk.DPU_EW_CFG_EW_OP_SRC__MASK) |
        self.reg(0, rk.DPU_EW_CFG_EW_MUL_PRELU__SHIFT, rk.DPU_EW_CFG_EW_MUL_PRELU__MASK) |
        self.reg(1, rk.DPU_EW_CFG_EW_OP_TYPE__SHIFT, rk.DPU_EW_CFG_EW_OP_TYPE__MASK) |
        self.reg(0, rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_BYPASS__MASK) |
        self.reg(0, rk.DPU_EW_CFG_EW_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_BYPASS__MASK))
    elif op in self.code_for_op.keys():
      self.emit_raw(rk.DPU, rk.REG_DPU_EW_CFG,
        self.reg(0, rk.DPU_EW_CFG_EW_CVT_TYPE__SHIFT, rk.DPU_EW_CFG_EW_CVT_TYPE__MASK) |
        self.reg(0, rk.DPU_EW_CFG_EW_CVT_ROUND__SHIFT, rk.DPU_EW_CFG_EW_CVT_ROUND__MASK) |
        self.reg(1, rk.DPU_EW_CFG_EW_DATA_MODE__SHIFT, rk.DPU_EW_CFG_EW_DATA_MODE__MASK) |
        self.reg(self.get_edata_size(dtype), rk.DPU_EW_CFG_EDATA_SIZE__SHIFT, rk.DPU_EW_CFG_EDATA_SIZE__MASK) |
        self.reg(0, rk.DPU_EW_CFG_EW_EQUAL_EN__SHIFT, rk.DPU_EW_CFG_EW_EQUAL_EN__MASK) |
        self.reg(0, rk.DPU_EW_CFG_EW_BINARY_EN__SHIFT, rk.DPU_EW_CFG_EW_BINARY_EN__MASK) |
        self.reg(self.code_for_op[op], rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT, rk.DPU_EW_CFG_EW_ALU_ALGO__MASK) |
        self.reg(0, rk.DPU_EW_CFG_EW_RELUX_EN__SHIFT, rk.DPU_EW_CFG_EW_RELUX_EN__MASK) |
        self.reg(1, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
        self.reg(0, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK) |
        self.reg(1, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
        self.reg(1, rk.DPU_EW_CFG_EW_OP_SRC__SHIFT, rk.DPU_EW_CFG_EW_OP_SRC__MASK) |
        self.reg(0, rk.DPU_EW_CFG_EW_MUL_PRELU__SHIFT, rk.DPU_EW_CFG_EW_MUL_PRELU__MASK) |
        self.reg(0, rk.DPU_EW_CFG_EW_OP_TYPE__SHIFT, rk.DPU_EW_CFG_EW_OP_TYPE__MASK) |
        self.reg(0, rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_BYPASS__MASK) |
        self.reg(0, rk.DPU_EW_CFG_EW_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_BYPASS__MASK))
  
  def create_channel(self, channel):
    self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
      self.reg(channel, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__MASK) |
      self.reg(channel, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))

    self.emit_raw(rk.DPU, rk.REG_DPU_WDMA_SIZE_0,
      self.reg(channel, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__SHIFT, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK))
    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
      self.reg(channel, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK))

  def create_size(self, height, width):
    self.emit_raw(rk.DPU, rk.REG_DPU_WDMA_SIZE_1,
      self.reg(height, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__MASK) |
      # self.reg(width, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))
      self.reg(0, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))
    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
      self.reg(width, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK))
    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
      self.reg(height, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__MASK))
  def create_stride(self, stride):
    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_SURF_STRIDE,
      self.reg(stride, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__SHIFT, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__MASK))
    self.emit_raw(rk.DPU, rk.REG_DPU_SURFACE_ADD,
      self.reg(stride, rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT, rk.DPU_SURFACE_ADD_SURF_ADD__MASK))
  def create_surf_notch(self, notch):
    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SURF_NOTCH,
      self.reg(notch, rk.DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR__SHIFT, rk.DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR__MASK))
    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_SURF_NOTCH,
      self.reg(notch, rk.DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH__SHIFT, rk.DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH__MASK))

  def create_reg(self):
    self.q = []
    self.emit_raw(rk.DPU, rk.REG_DPU_S_POINTER,
      self.reg(1  , rk.DPU_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_S_POINTER_POINTER_PP_MODE__MASK) |
      self.reg(1, rk.DPU_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_EN__MASK) |
      self.reg(1, rk.DPU_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_S_POINTER_POINTER_PP_EN__MASK))

    self.emit_raw(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
      self.reg(0xF, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      self.reg(0, rk.DPU_FEATURE_MODE_CFG_CONV_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_CONV_MODE__MASK) |
      self.reg(0x2, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK) |
      self.reg(0x1, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK))

    # Skip Transpose
    self.emit_raw(rk.DPU, rk.REG_DPU_BS_OW_CFG,
      # self.reg(3, rk.DPU_BS_OW_CFG_SIZE_E_2__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_2__MASK) |
      # self.reg(3, rk.DPU_BS_OW_CFG_SIZE_E_1__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_1__MASK) |
      # self.reg(3, rk.DPU_BS_OW_CFG_SIZE_E_0__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_0__MASK) |
      self.reg(1, rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT, rk.DPU_BS_OW_CFG_OD_BYPASS__MASK))
    # Skip Transpose
    self.emit_raw(rk.DPU, rk.REG_DPU_BS_OW_OP,
      self.reg(0, rk.DPU_BS_OW_OP_OW_OP__SHIFT, rk.DPU_BS_OW_OP_OW_OP__MASK))

    self.create_channel(7)
    self.create_size(0, 9)
    self.create_stride(12)
    self.create_surf_notch(2)

    self.emit_raw(rk.DPU, 0x40c4, 0);

    self.emit_raw(rk.DPU, rk.REG_DPU_LUT_ACCESS_CFG, 0);
    self.emit_raw(rk.DPU, rk.REG_DPU_LUT_ACCESS_DATA, 0);
    self.emit_raw(rk.DPU, rk.REG_DPU_LUT_CFG, 0);
    self.emit_raw(rk.DPU, rk.REG_DPU_LUT_INFO, 0);
    self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_START, 0);
    self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_END, 0);
    self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LO_START, 0);
    self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LO_END, 0);
    self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_SLOPE_SCALE, 0);
    self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LE_SLOPE_SHIFT, 0);
    self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LO_SLOPE_SCALE, 0);
    self.emit_raw(rk.DPU, rk.REG_DPU_LUT_LO_SLOPE_SHIFT, 0);

    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_BRDMA_CFG,
      self.reg(0, rk.DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE__SHIFT, rk.DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE__MASK))
    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_NRDMA_CFG, 0);
    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_BN_BASE_ADDR, 0);

    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_DMA_CFG, 0);
    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_PAD_CFG, 0);
    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT,
    self.reg(1, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__MASK) |
    self.reg(1, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__MASK) |
    self.reg(1, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__MASK) |
    self.reg(1, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__MASK))
    self.emit_raw(rk.DPU, rk.REG_DPU_BN_ALU_CFG,
      self.reg(0,0,0))
    self.emit_raw(rk.DPU, rk.REG_DPU_BN_MUL_CFG,
      self.reg(0,0,0))
    self.emit_raw(rk.DPU, rk.REG_DPU_BN_RELUX_CMP_VALUE,
      self.reg(0,0,0))
    self.emit_raw(rk.DPU, rk.REG_DPU_EW_CVT_OFFSET_VALUE,
      self.reg(0,0,0))

      # if float REG_DPU_EW_CVT_SCALE_VALUE = 0, else REG_DPU_EW_CVT_SCALE_VALUE = 1
    self.emit_raw(rk.DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE,
      self.reg(0, rk.DPU_EW_CVT_SCALE_VALUE_EW_TRUNCATE__SHIFT, rk.DPU_EW_CVT_SCALE_VALUE_EW_TRUNCATE__MASK) |
      self.reg(0, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SHIFT__SHIFT, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SHIFT__MASK) |
      self.reg(1, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__SHIFT, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__MASK))


    self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
      # self.reg(9, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
      self.reg(0, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))

    self.emit_raw(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE,
      self.reg(12, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK))

    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_BS_BASE_ADDR,0)
    
    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER,
      self.reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__MASK) |
      self.reg(1, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__MASK) |
      self.reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__MASK))
     

    self.emit_raw(rk.DPU, rk.REG_DPU_EW_RELUX_CMP_VALUE,
      self.reg(0,0,0))
    self.emit_raw(rk.DPU, rk.REG_DPU_OUT_CVT_OFFSET,
      self.reg(0,0,0))


   
    self.emit_raw(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT,
      self.reg(1-1, rk.DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT__SHIFT, rk.DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT__MASK))
    self.emit_raw(rk.DPU, rk.REG_DPU_EW_OP_VALUE_0, 0);
    self.emit_raw(rk.DPU, rk.REG_DPU_EW_OP_VALUE_1, 0);
    self.emit_raw(rk.DPU, rk.REG_DPU_EW_OP_VALUE_2, 0);
    self.emit_raw(rk.DPU, rk.REG_DPU_EW_OP_VALUE_3, 0);
    self.emit_raw(rk.DPU, rk.REG_DPU_EW_OP_VALUE_4, 0);
    self.emit_raw(rk.DPU, rk.REG_DPU_EW_OP_VALUE_5, 0);
    self.emit_raw(rk.DPU, rk.REG_DPU_EW_OP_VALUE_6, 0);
    self.emit_raw(rk.DPU, rk.REG_DPU_EW_OP_VALUE_7, 0);
 
  def submit(self):
    #self.q.append(0x2001000178495044), # 63
    self.emit_raw(0x00, 0x00, 0);
    self.emit_raw(rk.DPU, rk.REG_PC_REGISTER_AMOUNTS, 0);  
    self.q.append(0x0101000000000014);
    self.emit_raw(0x80, rk.REG_PC_OPERATION_ENABLE,
      self.reg(12, rk.PC_OPERATION_ENABLE_RESERVED_0__SHIFT, rk.PC_OPERATION_ENABLE_RESERVED_0__MASK) |
      self.reg(0, rk.PC_OPERATION_ENABLE_OP_EN__SHIFT, rk.PC_OPERATION_ENABLE_OP_EN__MASK))

    tasks = ctypes.cast(self.device.task_buf.va_addr, ctypes.POINTER(rk.struct_rknpu_task* 128)).contents
    regcmd = ctypes.cast(self.device.cmd_buf.va_addr, ctypes.POINTER(ctypes.c_uint64 * 128)).contents
    for i in range(len(self.q)):
      regcmd[i] = self.q[i]

    tasks[0].flags  = 0;
    tasks[0].op_idx = 4;
    tasks[0].enable_mask = 0x18;
    tasks[0].int_mask = 0x300;
    tasks[0].int_clear = 0x1ffff;
    tasks[0].int_status = 0;
    tasks[0].regcfg_amount = len(self.q)
    tasks[0].regcfg_offset = 0;
    tasks[0].regcmd_addr = self.device.cmd_buf.meta.dma_addr

    submit_res = rk.struct_rknpu_submit(
            flags=rk.RKNPU_JOB_PC | rk.RKNPU_JOB_BLOCK | rk.RKNPU_JOB_PINGPONG,
            timeout=6000,
            task_start=0,
            task_number=1,
            task_counter=0,
            priority=0,
            task_obj_addr=self.device.task_buf.meta.obj_addr,   # Placeholder, would be actual address in real code
            regcfg_obj_addr=0,
            task_base_addr=0,
            user_data=0,
            core_mask=1,
            fence_fd=-1,  
            subcore_task=(rk.struct_rknpu_subcore_task * 5)(
                rk.struct_rknpu_subcore_task(task_start=0, task_number=1),
                rk.struct_rknpu_subcore_task(task_start=1, task_number=0),
                rk.struct_rknpu_subcore_task(task_start=2, task_number=0),
            )
    )

    res = rk.DRM_IOCTL_RKNPU_SUBMIT(self.device.fd_ctl,   
            __payload=submit_res
    )

  def __init__(self, dev:RockchipDevice, name:str, lib:bytes):
    loaded = pickle.loads(lib)
    if isinstance(loaded, list):
      self.uops: list[tuple[Ops, DType|None, list[int], Any]] = loaded
      self._rk_conv_payload: tuple[Any, ...]|None = None
      if DEBUG >= 3:
        print("RockchipProgram uops sample", self.uops[:15])
    else:
      self.uops = []
      self._rk_conv_payload = loaded
    self.device = dev
    self.q = []
    self.code_for_op = RockchipRenderer.code_for_op
    if DEBUG >= 3:
      print("RockchipProgram init payload", type(loaded))



  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    if self._rk_conv_payload is not None:
      return self._execute_rk_conv(bufs, wait=wait)
    st = time.perf_counter()
    warp = list(itertools.product(*[range(x) for x in local_size[::-1]]))
    warp_size = len(warp)
    for idxs in itertools.product(*[range(x) for x in global_size[::-1]]):
      ul: dict[int, Any] = {}
      dl: dict[int, DType] = {}
      pbufs: list[memoryview] = list(bufs)
      pvals: list[int] = list(vals)
      i = 0
      loop_ends: dict[int, int] = {}
      while i < len(self.uops):
        uop, dtype, idp, arg = self.uops[i]
        void_ops = {Ops.ENDRANGE, Ops.BARRIER, Ops.IF, Ops.ENDIF, Ops.SINK, Ops.NOOP, Ops.STORE}
        inp = [ul[v] for v in idp if self.uops[v][0] not in void_ops]
        dtp = [dl[v] for v in idp if self.uops[v][0] not in void_ops]
        if getenv("TRACE"): print(i, uop, dtype, arg, inp, dtp)
        if uop is Ops.ENDRANGE:
          loop_ends[idp[0]] = i
          i = idp[0]
          continue
        if uop in (Ops.BARRIER, Ops.IF, Ops.ENDIF, Ops.SINK, Ops.NOOP):
          # in the python emulator, the warp is always in sync
          i += 1
          continue
        assert dtype is not None, f"{uop} is missing a dtype"
        dl[i] = dtype
        if uop is Ops.STORE:
          for j,val in enumerate(inp[1] if dtp[1].count > 1 else [inp[1]]):
            for (m,o,g),v in zip(inp[0], val):
              if g: _store(m, o+j, v, dtp[1].scalar())
          i += 1
          continue
        if uop in {Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL, Ops.DEFINE_REG}:
          assert isinstance(dtype, PtrDType), dtype
          storage_fmt = storage_fmt_for_dtype(dtype.base.scalar())
          if storage_fmt is None: raise RuntimeError(f"{dtype=} is not supported")
          if TYPE_CHECKING or sys.version_info < (3, 12): assert storage_fmt != "e"
          if uop is Ops.DEFINE_REG:
            # REGs are per thread
            ul[i] = [memoryview(bytearray(dtype.size*dtype.itemsize)).cast(storage_fmt) for _ in range(warp_size)]
          else:
            buf = memoryview(bytearray(dtype.size*dtype.itemsize)) if uop is not Ops.DEFINE_GLOBAL else pbufs.pop(0)
            ul[i] = [buf.cast(storage_fmt)] * warp_size
        elif uop is Ops.DEFINE_VAR:
          ul[i] = [pvals.pop(0)] * warp_size
        elif uop is Ops.SPECIAL:
          if arg[0] == 'g': ul[i] = [idxs[2-int(arg[-1])]] * warp_size
          elif arg[0] == 'l': ul[i] = [x[2-int(arg[-1])] for x in warp]
        elif uop is Ops.CONST: ul[i] = [arg] * warp_size
        elif uop is Ops.INDEX:
          ret:list = []
          if isinstance(dtp[0], ImageDType):
            for m,ox,oy in zip(inp[0], inp[1][0], inp[1][1]):
              if ox < 0 or ox >= dtp[0].shape[1] or oy < 0 or oy >= dtp[0].shape[0]: ret.append((m, None))
              else: ret.append((m, ox*4 + oy*dtp[0].shape[1]*4))
          else:
            for m,o in zip(inp[0], inp[1]): ret.append((m,o))
          ul[i] = [(m,o,g) for (m,o),g in zip(ret, inp[2] if len(inp) == 3 else [True]*len(ret))] # set the gate last
        elif uop is Ops.CAST and isinstance(dtype, PtrDType):
          ul[i] = inp[0]
        elif uop is Ops.RANGE:
          if i not in ul: ul[i] = [0] * warp_size
          else:
            for j in range(len(ul[i])):
              ul[i][j] += 1
            if ul[i][0] == inp[0][0]:
              del ul[i]
              i = loop_ends[i] + 1
              continue
        elif uop is Ops.VECTORIZE: ul[i] = inp
        elif uop is Ops.BITCAST:
          packed = struct.pack(str(warp_size) + storage_fmt_for_dtype(dtp[0].scalar()), *[to_storage_scalar(x, dtp[0].scalar()) for x in inp[0]])
          ul[i] = list(struct.unpack(str(warp_size) +  storage_fmt_for_dtype(dtype.scalar()), packed))
          ul[i] = [from_storage_scalar(x, dtype.scalar()) for x in ul[i]]
        elif uop is Ops.CAST:
          ul[i] = [truncate.get(dtype, lambda dt: dt)(dtypes.as_const(x, dtype)) for x in inp[0]]
        elif uop is Ops.LOAD:
          if dtype.count > 1:
            ul[i] = [load([inp[i][j] if i != 0 and dtp[i].count > 1 else inp[i] for i in range(len(inp))], j, dtype.scalar()) \
              for j in range(dtype.count)]
          else:
            ul[i] = load(inp, 0, dtype)
        elif uop is Ops.GEP: ul[i] = inp[0][get_single_element(arg)]

        elif uop is Ops.CUSTOM:
          print("Ops.CUSTOM in interpreter path")
          if isinstance(arg, RockchipConvInfo):
            raise RuntimeError("Unexpected RockchipConvInfo CUSTOM in interpreter path")
          lengths = [len(arr) for arr in inp]
          if not lengths:
            ul[i] = []
          else:
            max_len = max(lengths)
            broadcasted:list[list[Any]] = []
            for arr in inp:
              if len(arr) == max_len:
                broadcasted.append(arr)
              elif len(arr) == 1:
                broadcasted.append([arr[0]] * max_len)
              else:
                raise RuntimeError(f"broadcast mismatch for custom op: lengths={lengths}")
            ul[i] = [exec_alu(Ops.MUL, dtype, tuple(vals)) for vals in zip(*broadcasted)]
          i += 1
          continue
        elif uop in GroupOp.ALU:
          assert all_same([len(x) for x in inp]), f"{[len(x) for x in inp]} doesn't match on {uop}"
          assert all_same([dtype] + dtp) or uop in {Ops.CMPNE, Ops.CMPLT, Ops.WHERE}, f"dtype mismatch on {uop}"

          if (len(inp) == 2 
            and (dtype in (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int, dtypes.float, dtypes.float16))
            and (uop in RockchipRenderer.code_for_op.keys())):

   
            self.device.add_buffer(len(inp[0]))

            self.input_buf = self.device.input_buf
            self.weight_buf = self.device.weight_buf
            self.output_buf = self.device.output_buf

         
            import numpy as np
            self.create_reg()
            if dtype == dtypes.float or dtype == dtypes.float16:
              src = memoryview(bytearray(np.float16(inp[0]).tobytes()))
              ctypes.memmove(self.input_buf.va_addr, mv_address(src), src.nbytes)
              src2 = memoryview(bytearray(np.float16(inp[1]).tobytes()))
              ctypes.memmove(self.weight_buf.va_addr, mv_address(src2), src2.nbytes)
              # FIX ME
              dst = np.frombuffer((bytearray(self.output_buf.size * dtypes.float16.itemsize)), dtype=np.float16)
              # dst = np.frombuffer((bytearray(self.output_buf.size * dtypes.float32.itemsize)), dtype=np.float32)
              
              self.ops(uop, dtypes.float16)
   
            elif dtype == dtypes.int32 or dtype == dtypes.int16:
              src = memoryview(bytearray(np.int16(inp[0]).tobytes()))
              ctypes.memmove(self.input_buf.va_addr, mv_address(src), src.nbytes)
              src2 = memoryview(bytearray(np.int16(inp[1]).tobytes()))
              ctypes.memmove(self.weight_buf.va_addr, mv_address(src2), src2.nbytes)
              dst = np.frombuffer((bytearray(self.output_buf.size * dtypes.int16.itemsize)), dtype=np.int16)

              self.ops(uop, dtypes.int16)

            elif dtype == dtypes.int8:
              src = memoryview(bytearray(np.int8(inp[0]).tobytes()))
              ctypes.memmove(self.input_buf.va_addr, mv_address(src), src.nbytes)
              src2 = memoryview(bytearray(np.int8(inp[1]).tobytes()))
              ctypes.memmove(self.weight_buf.va_addr, mv_address(src2), src2.nbytes)
              dst = np.frombuffer((bytearray(self.output_buf.size * dtype.itemsize)), dtype=np.int8)

              self.ops(uop, dtypes.int8)

            self.emit_raw(rk.DPU, rk.REG_DPU_DST_BASE_ADDR, 
                self.reg(self.output_buf.meta.dma_addr, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
            self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
              self.reg(self.input_buf.meta.dma_addr, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK))
            self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
              self.reg(self.weight_buf.meta.dma_addr, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__MASK))
          
            self.submit()
            ctypes.memmove(dst.ctypes.data, self.output_buf.va_addr, self.output_buf.size * dtype.itemsize)
            # print("inp[0]", inp[0])            
            # print(uop)
            # print("inp[1]", inp[1])
            # print("dst", dst.tolist())
            ul[i] = dst.tolist()
          else:
            # CMPNE AND OR could be supported by NPU, need test
            if uop in (Ops.WHERE, Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE, Ops.XOR, Ops.AND, Ops.OR, Ops.TRUNC):
              if DEBUG >= 3:
                print('ALLOWED FALLBACK TO CPU', uop, dtype)
              ul[i] = [exec_alu(uop, dtype, p) for p in zip(*inp)]
            else:
              print('EXIT OPERATION NOT SUPPORTED', uop, dtype)
              exit()
        assert i in ul, (uop, dtype, idp, arg)
        i += 1
    return time.perf_counter() - st

  def _buffer_as_bytes(self, buf: Any) -> bytes:
    if isinstance(buf, (bytes, bytearray)): return bytes(buf)
    if isinstance(buf, memoryview): return buf.tobytes()
    if isinstance(buf, np.ndarray): return buf.tobytes()
    if hasattr(buf, "va_addr") and hasattr(buf, "size"):
      return ctypes.string_at(buf.va_addr, buf.size)
    raise TypeError(f"unsupported buffer type {type(buf)}")

  def _write_bytes(self, buf: Any, data: bytes) -> None:
    if isinstance(buf, bytearray):
      buf[:len(data)] = data
    elif isinstance(buf, memoryview):
      buf[:len(data)] = data
    elif hasattr(buf, "va_addr"):
      ctypes.memmove(buf.va_addr, data, len(data))
    else:
      raise TypeError(f"unsupported buffer type {type(buf)}")

  def _execute_rk_conv(self, bufs: tuple[Any, ...], wait: bool=False):
    assert self._rk_conv_payload is not None
    tag, dtype, info, metadata = self._rk_conv_payload
    assert tag == "RK_CONV"
    lhs_index = metadata["globals_order"].index(metadata["lhs"])
    rhs_index = metadata["globals_order"].index(metadata["rhs"])
    out_index = metadata["globals_order"].index(metadata["out"])
    lhs_buf = bufs[lhs_index]
    rhs_buf = bufs[rhs_index]
    out_buf = bufs[out_index]
    lhs_shape = info.lhs_tensor_shape or info.lhs_base_shape or info.lhs_shape
    rhs_shape = info.rhs_tensor_shape or info.rhs_base_shape or info.rhs_shape
    out_shape = info.out_tensor_shape or info.out_base_shape or info.out_shape
    def _np_dtype(dt: DType):
      if hasattr(dt, "np"): return dt.np
      mapping = {
        dtypes.float: np.float32,
        dtypes.float32: np.float32,
        dtypes.float16: np.float16,
        dtypes.bfloat16: np.float16,
        dtypes.int8: np.int8,
        dtypes.int16: np.int16,
        dtypes.int32: np.int32,
      }
      return mapping.get(dt, np.float32)
    if DEBUG:
      print("RK_CONV payload", info, metadata)
      print("lhs shape", lhs_shape, "rhs shape", rhs_shape, "out shape", out_shape)
    np_dtype = _np_dtype(dtype)
    post_ops = info.post_ops

    def _apply_post_ops(arr: np.ndarray, work_dtype: np.dtype) -> np.ndarray:
      result = arr
      for op, value in post_ops:
        if op is Ops.ADD:
          result = result + np.array(value, dtype=work_dtype)
        else:
          raise RuntimeError(f"Unsupported RK_CONV post-op: {op}")
      return result

    if len(lhs_shape) == 1 and len(rhs_shape) == 1:
      lhs_vec = np.frombuffer(self._buffer_as_bytes(lhs_buf), dtype=np_dtype)
      rhs_vec = np.frombuffer(self._buffer_as_bytes(rhs_buf), dtype=np_dtype)
      out_len = len(lhs_vec) - len(rhs_vec) + 1
      if out_len <= 0:
        raise RuntimeError("invalid 1D convolution dimensions")
      acc_dtype = np.float32 if np_dtype in (np.float16, np.float32) else np_dtype
      out_vec = np.zeros(out_len, dtype=acc_dtype)
      for i in range(out_len):
        out_vec[i] = np.sum(lhs_vec[i:i+len(rhs_vec)].astype(acc_dtype) * rhs_vec.astype(acc_dtype))
      if post_ops:
        out_vec = _apply_post_ops(out_vec, acc_dtype)
      if acc_dtype != np_dtype:
        out_vec = out_vec.astype(np_dtype)
      total_elems = int(np.prod(out_shape)) if out_shape else out_len
      if total_elems == out_len and out_shape:
        reshaped = out_vec.reshape(out_shape)
      else:
        reshaped = out_vec.reshape((out_len,))
      self._write_bytes(out_buf, reshaped.tobytes())
      return 0.0

    if len(lhs_shape) != 4 or len(rhs_shape) != 4 or len(out_shape) != 4:
      raise RuntimeError("RK_CONV fast path not handled")

    lhs_arr = np.frombuffer(self._buffer_as_bytes(lhs_buf), dtype=np_dtype).reshape(lhs_shape)
    rhs_arr = np.frombuffer(self._buffer_as_bytes(rhs_buf), dtype=np_dtype).reshape(rhs_shape)

    N, C, H, W = lhs_shape
    K, Cw, KH, KW = rhs_shape
    assert Cw == C, "channel mismatch"
    outH = H - KH + 1
    outW = W - KW + 1
    if outH <= 0 or outW <= 0:
      raise RuntimeError("invalid convolution dimensions")

    acc_dtype = np.float32 if np_dtype in (np.float16, np.float32) else np_dtype
    out_arr = np.zeros((N, K, outH, outW), dtype=acc_dtype)
    for n in range(N):
      for k in range(K):
        for c in range(C):
          for y in range(outH):
            for x in range(outW):
              window = lhs_arr[n, c, y:y+KH, x:x+KW]
              kernel = rhs_arr[k, c]
              out_arr[n, k, y, x] += np.sum(window.astype(acc_dtype) * kernel.astype(acc_dtype))

    if post_ops:
      out_arr = _apply_post_ops(out_arr, acc_dtype)

    if acc_dtype != np_dtype:
      out_arr = out_arr.astype(np_dtype)
    out_arr = out_arr.reshape(out_shape)
    self._write_bytes(out_buf, out_arr.tobytes())
    return 0.0

class RockchipRegisterAllocator(HCQAllocatorBase):
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    return self.dev._gpu_alloc(size, 0)
  def _do_copy(self, src_addr, dest_addr, src_size):
    ctypes.memmove(dest_addr, src_addr, src_size)

  def _copyin(self, dest:HCQBuffer, src:memoryview):
    self._do_copy(mv_address(src), dest.va_addr, src.nbytes)

  def _copyout(self, dest:memoryview, src:HCQBuffer):
    self._do_copy(src.va_addr, mv_address(dest), src.size)

  def _as_buffer(self, src:HCQBuffer) -> memoryview:
    return to_mv(ctypes.cast(int, src.va_addr), src.size)

class RockchipAllocator(Allocator['RockchipDevice']):
  def _alloc(self, size, options): return memoryview(bytearray(size))
  def _copyin(self, dest, src:memoryview): dest[:] = src
  def _copyout(self, dest:memoryview, src): dest[:] = src

class RockchipCompiler(Compiler):
  def compile(self, src:str) -> bytes: return base64.b64decode(src)
