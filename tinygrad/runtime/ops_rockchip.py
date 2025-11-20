# pylint: disable=cell-var-from-loop
# a python uops emulator
# works to test the tensor cores, and all the uops in general
# this is the (living) definition of uops
import array
import ctypes
import functools
import mmap
import os
import math
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

_active_conv_metadata: tuple[str, ...]|None = None
_active_conv_metadata_used: bool = False

def set_active_conv_metadata(names:tuple[str, ...]|None):
  global _active_conv_metadata, _active_conv_metadata_used
  _active_conv_metadata = names
  _active_conv_metadata_used = False

def _metadata_names(uop:UOp) -> tuple[str, ...]:
  return tuple(m.name for m in (uop.metadata or ())) if hasattr(uop, "metadata") else tuple()

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
          if len(candidate) == 1: score -= 1
          if score > best_score:
            best_score = score
            best = candidate
      except Exception:
        continue
  if best is not None: return best
  if base_shape: return base_shape
  return _safe_shape(u)

MOVEMENT_OPS = {Ops.RESHAPE, Ops.PERMUTE, Ops.SHRINK, Ops.EXPAND, Ops.VIEW, Ops.CONTIGUOUS, Ops.CAST, Ops.BITCAST}
TERMINAL_OPS = {Ops.STORE, Ops.SINK, Ops.ASSIGN, Ops.KERNEL}
FUSIBLE_OPS = {Ops.ADD} if FUSE_POSTOPS else set()

def _extract_const(node:UOp) -> Any|None:
  cur = node
  seen:set[UOp] = set()
  while True:
    if cur in seen: return None
    seen.add(cur)
    if cur.op is Ops.CONST: return cur.arg
    if cur.op in MOVEMENT_OPS and len(cur.src) == 1:
      cur = cur.src[0]
      continue
    return None

def _depends_on(node:UOp, target:UOp) -> bool:
  stack:list[UOp] = [node]
  seen:set[UOp] = set()
  while stack:
    cur = stack.pop()
    if cur is target: return True
    if cur in seen: continue
    seen.add(cur)
    stack.extend(cur.src)
  return False

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
      if child.op in MOVEMENT_OPS:
        stack.append(child)
        continue
      if child.op in FUSIBLE_OPS:
        conv_operand:UOp|None = None
        const_operand:UOp|None = None
        for src in child.src:
          if _depends_on(src, root):
            if conv_operand is not None:
              return False, []
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
      if child.op in TERMINAL_OPS:
        continue
      return False, []
  return True, post

def _build_conv_info(meta_names:tuple[str, ...], axes:tuple[int, ...],
                     lhs:UOp, rhs:UOp, out_node:UOp,
                     post_ops:list[tuple[Ops, Any]],
                     shape_overrides:dict[str, tuple[int, ...]]|None=None) -> RockchipConvInfo:
  _bs_lhs = _base_shape(lhs)
  _bs_rhs = _base_shape(rhs)
  _bs_out = _base_shape(out_node)
  lhs_tensor_shape = shape_overrides.get("lhs_tensor_shape") if shape_overrides else None
  rhs_tensor_shape = shape_overrides.get("rhs_tensor_shape") if shape_overrides else None
  out_tensor_shape = shape_overrides.get("out_tensor_shape") if shape_overrides else None
  return RockchipConvInfo(
    meta_names, axes,
    _safe_shape(lhs), _safe_shape(rhs), _safe_shape(out_node),
    _bs_lhs, _bs_rhs, _bs_out,
    lhs_tensor_shape if lhs_tensor_shape is not None else _tensor_shape(lhs, _bs_lhs),
    rhs_tensor_shape if rhs_tensor_shape is not None else _tensor_shape(rhs, _bs_rhs),
    out_tensor_shape if out_tensor_shape is not None else _tensor_shape(out_node, _bs_out),
    tuple(post_ops))

def _has_reduce_descendant(node:UOp) -> bool:
  seen:set[UOp] = set()
  stack:list[UOp] = [node]
  while stack:
    cur = stack.pop()
    for ref in list(cur.children):
      child = ref()
      if child is None or child in seen: continue
      if child.op is Ops.REDUCE_AXIS:
        return True
      seen.add(child)
      stack.append(child)
  return False

def _axes_from_metadata(lhs:UOp, meta_name:str) -> tuple[int, ...]:
  try:
    _, data = _parse_conv_metadata(meta_name)
    hw = data.get("hw", tuple())
    lhs_shape = _safe_shape(lhs)
    if hw and len(lhs_shape) >= len(hw):
      return tuple(range(len(lhs_shape) - len(hw), len(lhs_shape)))
  except Exception:
    return tuple()
  return tuple()

def _conv_shape_overrides(meta_name:str) -> dict[str, tuple[int, ...]]:
  try:
    _, data = _parse_conv_metadata(meta_name)
  except Exception:
    return {}
  overrides: dict[str, tuple[int, ...]] = {}
  for key, target in (("lhs", "lhs_tensor_shape"), ("rhs", "rhs_tensor_shape"), ("out", "out_tensor_shape")):
    if (val:=data.get(key)):
      overrides[target] = tuple(int(x) for x in val)
  return overrides

def _rockchip_conv_rewrite(red:UOp) -> UOp|None:
  if len(red.src) != 1: return None
  mul = _peel_mul(red.src[0])
  if mul is None or len(mul.src) != 2: return None
  lhs, rhs = mul.src
  meta_names = _metadata_names(red)
  if DEBUG >= 3:
    print("ROCKCHIP rewrite candidate", meta_names, red.arg, tuple(red.shape),
          lhs.op, tuple(getattr(lhs, "shape", ())), tuple(getattr(lhs, "full_shape", ())), tuple(getattr(lhs.base, "shape", ())) if hasattr(lhs, "base") else (),
          rhs.op, tuple(getattr(rhs, "shape", ())), tuple(getattr(rhs, "full_shape", ())), tuple(getattr(rhs.base, "shape", ())) if hasattr(rhs, "base") else ())
    if DEBUG >= 6:
      print("lhs metadata", getattr(lhs, "metadata", None))
      print("rhs metadata", getattr(rhs, "metadata", None))
      print("lhs parent shapes", [getattr(p, "shape", None) for p in lhs.src])
      print("rhs parent shapes", [getattr(p, "shape", None) for p in rhs.src])
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
  ok, post_ops = _downstream_info(red)
  if not ok: return None
  info = _build_conv_info(meta_names, axes, lhs, rhs, red, post_ops)
  if DEBUG >= 2:
    parent_ops = [p.op for p in red.parents]
    print("ROCKCHIP conv rewrite applied", info, "parents", parent_ops)
  return UOp(Ops.CUSTOM, red.dtype, src=mul.src, arg=info, metadata=red.metadata)

def _rockchip_conv_metadata_rewrite(node:UOp) -> UOp|None:
  global _active_conv_metadata_used
  meta_names = _metadata_names(node)
  conv_meta_name = next((name for name in meta_names if name.startswith("conv")), None)
  used_hint = False
  if conv_meta_name is None and not meta_names and _active_conv_metadata and not _active_conv_metadata_used:
    meta_names = _active_conv_metadata
    conv_meta_name = next((name for name in meta_names if name.startswith("conv")), None)
    used_hint = conv_meta_name is not None
  if conv_meta_name is None: return None
  mul = _peel_mul(node)
  if mul is None or len(mul.src) != 2: return None
  if _has_reduce_descendant(mul): return None
  lhs, rhs = mul.src
  axes = _axes_from_metadata(lhs, conv_meta_name)
  shape_overrides = _conv_shape_overrides(conv_meta_name)
  ok, post_ops = _downstream_info(node)
  if not ok: return None
  info = _build_conv_info(meta_names, axes, lhs, rhs, node, post_ops, shape_overrides)
  if used_hint:
    _active_conv_metadata_used = True
  if DEBUG >= 2:
    print("ROCKCHIP metadata rewrite applied", info)
  return UOp(Ops.CUSTOM, node.dtype, src=mul.src, arg=info, metadata=node.metadata)

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
    if rewritten is None:
      rewritten = _rockchip_conv_metadata_rewrite(u)
    if rewritten is not None and rewritten is not u:
      return _rewrite(rewritten)
    return u

  return _rewrite(ast)

def storage_fmt_for_dtype(dtype: DType):
  if dtype in (dtypes.bfloat16, dtypes.float16): return 'H'
  return dtype.fmt

def to_storage_scalar(x, dtype: DType):
  if dtype == dtypes.bfloat16: return (struct.unpack('I', struct.pack('f', float_to_bf16(x)))[0] >> 16) & 0xFFFF
  if dtype == dtypes.float16:
    return int(np.frombuffer(np.array([x], dtype=np.float16).tobytes(), dtype=np.uint16)[0])
  return x

def from_storage_scalar(x, dtype: DType):
  if dtype == dtypes.bfloat16: return struct.unpack('f', struct.pack('I', (x & 0xFFFF) << 16))[0]
  if dtype == dtypes.float16:
    return float(np.frombuffer(np.array([x], dtype=np.uint16).tobytes(), dtype=np.float16)[0])
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

def _parse_conv_metadata(name:str) -> tuple[str, dict[str, tuple[int, ...]]]:
  parts = name.split("|")
  base = parts[0]
  data: dict[str, tuple[int, ...]] = {}
  for part in parts[1:]:
    if "=" not in part: continue
    key, val = part.split("=", 1)
    if not val:
      data[key] = ()
      continue
    data[key] = tuple(int(x) for x in val.split(",") if x)
  return base, data


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
    if DEBUG >= 3:
      for u in uops:
        if u.metadata:
          print("RK_RENDER metadata", u.op, u.metadata)
    conv = next((u for u in uops if u.op is Ops.CUSTOM and isinstance(u.arg, RockchipConvInfo)), None)
    def _depends_on(node:UOp, target:UOp) -> bool:
      stack:list[UOp] = [node]
      seen:set[UOp] = set()
      while stack:
        current = stack.pop()
        if current is target: return True
        if current in seen: continue
        seen.add(current)
        stack.extend(current.src)
      return False
    def _find_store_for_conv(conv_node:UOp) -> UOp|None:
      for store in uops:
        if store.op is not Ops.STORE: continue
        if len(store.src) > 1 and _depends_on(store.src[1], conv_node):
          return store
      return None
    conv_info = None
    if conv is None:
      for u in uops:
        if not (metadata := getattr(u, "metadata", None)):
          continue
        for meta in metadata:
          if not meta.name.startswith("conv"):
            continue
          _, data = _parse_conv_metadata(meta.name)
          lhs_shape = data.get("lhs")
          rhs_shape = data.get("rhs")
          out_shape = data.get("out")
          if not lhs_shape or not rhs_shape or not out_shape:
            if DEBUG:
              print("RK_CONV render fallback: malformed metadata", meta.name)
            continue
          conv_info = RockchipConvInfo(
            tuple(m.name for m in metadata),
            tuple(),
            lhs_shape,
            rhs_shape,
            out_shape,
            lhs_shape,
            rhs_shape,
            out_shape,
            lhs_shape,
            rhs_shape,
            out_shape)
          conv = u
          break
        if conv is not None:
          break
    if DEBUG >= 3 and conv is not None and not isinstance(conv.arg, RockchipConvInfo):
      print("RK_CONV renderer metadata fallback used", conv_info)
    if conv is not None:
      info_value = conv.arg if isinstance(conv.arg, RockchipConvInfo) else conv_info
      if info_value is None:
        if DEBUG:
          print("RK_CONV render fallback: missing RockchipConvInfo", conv)
      else:
        def _find_global_id(node:UOp) -> int|None:
          for parent in node.toposort():
            if parent.op is Ops.DEFINE_GLOBAL:
              return parent.arg
          return None
        def _depends_on(node:UOp, target:UOp) -> bool:
          stack:list[UOp] = [node]
          seen:set[UOp] = set()
          while stack:
            current = stack.pop()
            if current is target: return True
            if current in seen: continue
            seen.add(current)
            stack.extend(current.src)
          return False
        store_uop = _find_store_for_conv(conv)
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
            payload = ("RK_CONV", conv.dtype, info_value, metadata)
            return base64.b64encode(pickle.dumps(payload)).decode()

    # the value of SPECIAL comes from local/global_size, not form its source
    lops = [(u.op, u.dtype, [uops.index(v) for v in u.src if u.op is not Ops.SPECIAL], u.arg) for u in uops]
    return base64.b64encode(pickle.dumps(lops)).decode()


class RockchipDevice(Compiled):
  def create_flink_name(self, handle: int, name:str, virt_address:int|None=None, dma_address:int|None=None) -> int:
    """
    Create a flink name for a GEM handle using DRM_IOCTL_GEM_FLINK.
    Args:
      handle: The GEM handle to create a flink name for
      name: Descriptive buffer name used for logging
      virt_address: Optional virtual address of the mapped buffer
      dma_address: Optional DMA/physical address of the buffer

    Returns:
      The flink name (uint32) on success, raises exception on failure
    """
    flink_req = rk.struct_drm_gem_flink(handle=handle, name=0)
    
    try:
      result = rk.DRM_IOCTL_GEM_FLINK(self.fd_ctl, __payload=flink_req)
      
      addr_info_parts = []
      if virt_address is not None: addr_info_parts.append(f"va {hex(virt_address)}")
      if dma_address is not None: addr_info_parts.append(f"dma {hex(dma_address)}")
      addr_info = f" {' '.join(addr_info_parts)}" if addr_info_parts else ""
      print(f"SUCCESS: Created flink name {flink_req.name} for handle {handle} {name}{addr_info}")
      return flink_req.name
    except Exception as e:
      print(f"ERROR: DRM_IOCTL_GEM_FLINK failed: {e}")
      raise

  def _gpu_alloc(self, size:int, flags, name:str) -> HCQBuffer:
    mem_create = rk.DRM_IOCTL_RKNPU_MEM_CREATE(self.fd_ctl, size=size, flags=flags | rk.RKNPU_MEM_NON_CACHEABLE)
    mem_map = rk.DRM_IOCTL_RKNPU_MEM_MAP(self.fd_ctl, handle=mem_create.handle, offset=0)
    va_addr = self.fd_ctl.mmap(0, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, mem_map.offset)

    flink_name = self.create_flink_name(mem_create.handle, name, virt_address=va_addr, dma_address=mem_create.dma_addr)
    mem_create.flink_name = flink_name

    return HCQBuffer(va_addr=va_addr, size=size, meta=mem_create)

  def _gpu_free(self, mem:HCQBuffer):
    if mem is None:
      return
    rk.DRM_IOCTL_RKNPU_MEM_DESTROY(self.fd_ctl, handle=mem.meta.handle, obj_addr=mem.meta.obj_addr, reserved=0)
    FileIOInterface.munmap(mem.va_addr, mem.size)

  def __init__(self, device:str): 
    self.fd_ctl = FileIOInterface(f"/dev/dri/card1", os.O_RDWR)
    self.task_buf = self._gpu_alloc(1024, rk.RKNPU_MEM_KERNEL_MAPPING, name="task")
    self.cmd_buf = self._gpu_alloc(8192, 0, name="cmd")

    self.input_buf = None
    self.weight_buf = None
    self.output_buf = None
    self._submission_total = 0

    self.buffer_list = []
    self.code_for_op = RockchipRenderer.code_for_op

    super().__init__(device, RockchipAllocator(self), RockchipRenderer(), RockchipCompiler(), functools.partial(RockchipProgram, self))

  def add_buffer(self, size):

    self.input_buf = next((item["buf"] for item in self.buffer_list if item["buf_type"] == "input" and item["size"] == size), None)
    self.weight_buf = next((item["buf"] for item in self.buffer_list if item["buf_type"] == "weight" and item["size"] == size), None)
    self.output_buf = next((item["buf"] for item in self.buffer_list if item["buf_type"] == "output" and item["size"] == size), None)
    if (self.input_buf is None or self.weight_buf is None or self.output_buf is None):
      self.input_buf = self._gpu_alloc(size, 0, name="input")
      self.buffer_list.append({"buf_type": "input", "buf": self.input_buf, "size": size})
      self.weight_buf = self._gpu_alloc(size, 0, name="weight")
      self.buffer_list.append({"buf_type": "weight", "buf": self.weight_buf, "size": size})
      self.output_buf = self._gpu_alloc(size, 0, name="output")
      self.buffer_list.append({"buf_type": "output", "buf": self.output_buf, "size": size})

  def reset_submission_count(self):
    self._submission_total = 0

  def submission_count(self) -> int:
    return self._submission_total

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
    if reset_queue:
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


    print("DRM_IOCTL_RKNPU_SUBMIT")
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

  def _submit_conv(self, cmd_sequences:list[list[int]]|None=None) -> None:
    sequences = cmd_sequences if cmd_sequences is not None else [list(self.q)]
    if not sequences:
      return
    self._submit_count = getattr(self, "_submit_count", 0) + 1
    if hasattr(self.device, "_submission_total"):
      self.device._submission_total += 1
    if DEBUG >= 2:
      print(f"RK_CONV submit count {self._submit_count}")
      conv_debug = getattr(self, "_rk_conv_debug", None)
      if conv_debug is not None:
        input_dma, weight_dma, output_dma = conv_debug["dma"]
        print(f"RK_CONV DMA input {input_dma:#x} weight {weight_dma:#x} output {output_dma:#x}")
        print(f"RK_CONV stride {conv_debug['dst_stride']} surface_add {conv_debug['surface_add']} batch {conv_debug['batch_count']}")
    tasks = ctypes.cast(self.device.task_buf.va_addr, ctypes.POINTER(rk.struct_rknpu_task * 128)).contents
    reg_entries = self.device.cmd_buf.size // ctypes.sizeof(ctypes.c_uint64)
    reg_array_type = ctypes.c_uint64 * reg_entries
    regcmd = ctypes.cast(self.device.cmd_buf.va_addr, ctypes.POINTER(reg_array_type)).contents
    offsets:list[int] = []
    used_entries = 0
    for seq in sequences:
      if used_entries + len(seq) > reg_entries:
        raise RuntimeError("RK_CONV command buffer overflow")
      offsets.append(used_entries)
      for idx, word in enumerate(seq):
        regcmd[used_entries + idx] = word
      used_entries += len(seq)
    for idx in range(used_entries, len(regcmd)):
      regcmd[idx] = 0

    task_entries = min(len(tasks), self.device.task_buf.size // ctypes.sizeof(rk.struct_rknpu_task))
    if task_entries == 0:
      raise RuntimeError("RK_CONV task buffer too small")
    if len(sequences) > task_entries:
      raise RuntimeError("RK_CONV task buffer overflow")
    for task_idx, (seq, offset) in enumerate(zip(sequences, offsets)):
      tasks[task_idx].flags = 0
      tasks[task_idx].op_idx = 0
      tasks[task_idx].enable_mask = 0xd
      tasks[task_idx].int_mask = 0x300
      tasks[task_idx].int_clear = 0x1ffff
      tasks[task_idx].int_status = 0x100
      tasks[task_idx].regcfg_amount = len(seq)
      tasks[task_idx].regcfg_offset = 0
      tasks[task_idx].regcmd_addr = self.device.cmd_buf.meta.dma_addr + offset * ctypes.sizeof(ctypes.c_uint64)
    for idx in range(len(sequences), task_entries):
      tasks[idx].flags = 0
      tasks[idx].op_idx = 0
      tasks[idx].enable_mask = 0
      tasks[idx].int_mask = 0
      tasks[idx].int_clear = 0
      tasks[idx].int_status = 0
      tasks[idx].regcfg_amount = 0
      tasks[idx].regcfg_offset = 0
      tasks[idx].regcmd_addr = 0

    submit_res = rk.struct_rknpu_submit(
      flags=rk.RKNPU_JOB_PC | rk.RKNPU_JOB_BLOCK | rk.RKNPU_JOB_PINGPONG,
      timeout=6000,
      task_start=0,
      task_number=len(sequences),
      task_counter=0,
      priority=0,
      task_obj_addr=self.device.task_buf.meta.obj_addr,
      regcfg_obj_addr=0,
      task_base_addr=0,
      user_data=0,
      core_mask=1,
      fence_fd=-1,
      subcore_task=(rk.struct_rknpu_subcore_task * 5)(
        rk.struct_rknpu_subcore_task(task_start=0, task_number=len(sequences)),
        rk.struct_rknpu_subcore_task(task_start=1, task_number=0),
        rk.struct_rknpu_subcore_task(task_start=2, task_number=0),
      ),
    )
    print("DRM_IOCTL_RKNPU_SUBMIT conv")
    # os.system("bash -c 'cd ~/npu/ops_reg/ && python dump.py 1' ")
    # os.system("bash -c 'cd ~/npu/ops_reg/ && python dump.py 2' ")
    # os.system("bash -c 'cd ~/npu/ops_reg/ && python dump.py 3' ")
    # os.system("bash -c 'cd ~/npu/ops_reg/ && python dump.py 4' ")

    rk.DRM_IOCTL_RKNPU_SUBMIT(self.device.fd_ctl, __payload=submit_res)
    # os.system("bash -c 'cd ~/npu/ops_reg/ && python dump.py 5' ")
    # os.system("bash -c 'cd ~/npu/ops_reg/ && python dump.py 6' ")

    self.q = []

  def _apply_post_ops_array(self, arr: np.ndarray, post_ops: tuple[tuple[Ops, Any], ...]) -> np.ndarray:
    if not post_ops: return arr
    result = arr
    for op, value in post_ops:
      if op is Ops.ADD:
        result = result + np.array(value, dtype=result.dtype)
      else:
        raise RuntimeError(f"Unsupported RK_CONV post-op: {op}")
    return result

  def _conv1d_shape_info(self, info:RockchipConvInfo, lhs_len_single:int, rhs_len_single:int) -> tuple[int, int, int, int, int, int, int]|None:
    conv_meta_name = next((name for name in info.metadata if name.startswith("conv")), None)
    if conv_meta_name is None: return None
    try:
      _, data = _parse_conv_metadata(conv_meta_name)
    except Exception:
      return None
    lhs_shape = data.get("lhs")
    rhs_shape = data.get("rhs")
    out_shape = data.get("out")
    hw_shape = data.get("hw")
    stride_shape = data.get("stride")
    dilation_shape = data.get("dilation")
    groups_shape = data.get("groups")
    if not lhs_shape or not rhs_shape or not out_shape:
      return None
    input_width = int(lhs_shape[-1])
    output_width = int(out_shape[-1])
    kernel_width = int(hw_shape[-1]) if hw_shape else max(1, input_width - output_width + 1)
    in_channels = int(rhs_shape[1]) if len(rhs_shape) > 1 else 1
    out_channels = int(rhs_shape[0]) if len(rhs_shape) > 0 else 1
    groups = int(groups_shape[0]) if groups_shape else 1
    stride = int(stride_shape[-1]) if stride_shape else 1
    dilation = int(dilation_shape[-1]) if dilation_shape else 1
    if input_width <= 0 or output_width <= 0 or kernel_width <= 0:
      return None
    if stride <= 0 or dilation <= 0 or groups <= 0:
      return None
    return (input_width, kernel_width, output_width, in_channels, out_channels, groups, stride, dilation)

  def _conv1d_hw_full(self, lhs_vec: np.ndarray, rhs_vec: np.ndarray, dtype: DType, np_dtype: np.dtype,
                      input_width:int, kernel_width:int, output_width:int,
                      in_channels:int, out_channels:int, groups:int, stride:int, dilation:int,
                      batch_count:int=1) -> np.ndarray|None:
    if any(x != 1 for x in (groups, stride, dilation)): return None
    if batch_count <= 0 or in_channels <= 0 or out_channels <= 0: return None
    if input_width < kernel_width: return None
    sample_elems = input_width * in_channels
    if lhs_vec.size != sample_elems * batch_count: return None
    if rhs_vec.size != out_channels * in_channels * kernel_width: return None

    lhs_fp16 = np.ascontiguousarray(lhs_vec.astype(np.float16, copy=False))
    rhs_fp16 = np.ascontiguousarray(rhs_vec.astype(np.float16, copy=False))

    channel_align = max(8, ((in_channels + 7) // 8) * 8)
    # Hardware packs conv1d outputs in NC1HWC2 with 8-channel blocks (matches RKNN dumps).
    out_channel_align = max(8, ((out_channels + 7) // 8) * 8)
    weight_channel_align = max(8, ((out_channels + 7) // 8) * 8)
    input_width_aligned = (input_width + 15) & ~15
    dst_stride = output_width
    element_size = np.dtype(np.float16).itemsize
    row_bytes = dst_stride * out_channel_align * element_size
    surface_add = dst_stride
    if DEBUG >= 2:
      print("RK_CONV layout dst_stride", dst_stride, "out_channel_align", out_channel_align,
            "row_bytes", row_bytes, "surface_add", surface_add, "batch", batch_count)

    lhs_view = lhs_fp16.reshape(batch_count, in_channels, 1, input_width)
    def _pack_sample(idx:int) -> np.ndarray|None:
      packed = self._pack_nc1hwc2_fp16(lhs_view[idx:idx+1], 1, in_channels, 1, input_width, 1, input_width)
      return packed if packed.size != 0 else None
    first_packed = _pack_sample(0)
    if first_packed is None:
      return None
    input_bytes = first_packed.nbytes

    packed_weights = self._pack_conv_weights_fp16(
      rhs_fp16.reshape(out_channels, in_channels, kernel_width),
      out_channels, in_channels, 1, kernel_width, channel_align, weight_channel_align)
    if packed_weights.size == 0:
      return None

    padded_kernel_bytes = kernel_width * channel_align * np.dtype(np.float16).itemsize
    output_c1 = (out_channels + out_channel_align - 1) // out_channel_align
    packed_output_elems_per_sample = output_c1 * dst_stride * out_channel_align
    weight_bytes = packed_weights.nbytes
    output_bytes = packed_output_elems_per_sample * element_size
    if DEBUG >= 2:
      print("RK_CONV packed_input elems", first_packed.size, "bytes", input_bytes,
            "width_stride", input_width_aligned, "channel_align", channel_align)
      print("RK_CONV packed_output elems", packed_output_elems_per_sample, "bytes", output_bytes,
            "dst_stride", dst_stride, "out_channel_align", out_channel_align)

    weight_hw = input_hw = output_hw = None
    weight_hw = self.device._gpu_alloc(weight_bytes, 0, name="rk_conv_weight")
    input_hw = self.device._gpu_alloc(input_bytes, 0, name="rk_conv_input")
    output_hw = self.device._gpu_alloc(output_bytes, 0, name="rk_conv_output")

    try:
      ctypes.memset(weight_hw.va_addr, 0, weight_bytes)
      ctypes.memmove(weight_hw.va_addr, packed_weights.tobytes(), weight_bytes)
      weight_dma = weight_hw.meta.dma_addr

      results = np.zeros((batch_count, out_channels, output_width), dtype=np.float32)
      first_packed_bytes = first_packed.tobytes()
      for batch_idx in range(batch_count):
        if batch_idx == 0:
          packed_bytes = first_packed_bytes
        else:
          packed_sample = _pack_sample(batch_idx)
          if packed_sample is None:
            return None
          packed_bytes = packed_sample.tobytes()

        ctypes.memset(input_hw.va_addr, 0, input_bytes)
        ctypes.memmove(input_hw.va_addr, packed_bytes, input_bytes)
        ctypes.memset(output_hw.va_addr, 0, output_bytes)

        self._program_conv1d_fp16(
          input_hw.meta.dma_addr, weight_dma, output_hw.meta.dma_addr,
          input_width, kernel_width, output_width,
          in_channels, out_channels,
          input_width_aligned, channel_align,
          out_channel_align, dst_stride,
          surface_add, padded_kernel_bytes,
          batch_count=1, reset_queue=True)
        self._submit_conv()

        packed_output = np.frombuffer(ctypes.string_at(output_hw.va_addr, output_bytes),
                                      dtype=np.float16, count=packed_output_elems_per_sample)
        unpacked = self._unpack_nc1hwc2_fp16(packed_output, 1, out_channels, 1, output_width,
                                             out_channel_align, dst_stride)
        results[batch_idx] = unpacked[0, :, 0, :]

      if DEBUG >= 2:
        print("RK_CONV unpacked sample", results[0])
        input_batch = lhs_view[0, :, 0, :].astype(np.float32)
        kernel_ref = rhs_fp16.reshape(out_channels, in_channels, kernel_width).astype(np.float32)
        cpu_ref = np.zeros((out_channels, output_width), dtype=np.float32)
        for oc in range(out_channels):
          for pos in range(output_width):
            acc = 0.0
            for ic in range(in_channels):
              for k in range(kernel_width):
                acc += float(input_batch[ic, pos + k]) * float(kernel_ref[oc, ic, k])
            cpu_ref[oc, pos] = acc
        diff = np.abs(results[0] - cpu_ref)
        if diff.size:
          oc_pos = np.unravel_index(int(np.argmax(diff)), diff.shape)
          max_diff = float(diff[oc_pos])
        else:
          oc_pos = (0, 0)
          max_diff = 0.0
        if max_diff > 1e-2:
          npu_val = float(results[0, oc_pos[0], oc_pos[1]])
          cpu_val = float(cpu_ref[oc_pos])
          print(f"RK_CONV CPU verify mismatch diff {max_diff:.6f} at oc={oc_pos[0]} pos={oc_pos[1]} npu={npu_val:.6f} cpu={cpu_val:.6f}")
      return results.reshape(batch_count * out_channels * output_width).astype(np_dtype, copy=False)
    finally:
      for buf in (input_hw, output_hw, weight_hw):
        if buf is not None and hasattr(self.device, "_gpu_free"):
          self.device._gpu_free(buf)

  def _unpack_nc1hwc2_fp16(self, src: np.ndarray, batch:int, channels:int, height:int, width:int,
                           c2:int, width_stride:int) -> np.ndarray:
    """
    Convert NC1HWC2 tensors back to NCHW order using the same layout walked by
    the RKNN sample in `npu/ops_rknn/dump/conv1d_i81_11_w611.h`.
    """
    if batch <= 0 or channels <= 0 or height <= 0 or width <= 0 or width_stride <= 0 or c2 <= 0:
      return np.zeros((max(batch, 0), max(channels, 0), max(height, 0), max(width, 0)), dtype=np.float32)

    c1 = max(1, (channels + c2 - 1) // c2)
    plane_stride = height * width_stride * c2
    dst = np.zeros((batch, channels, height, width), dtype=np.float32)
    src_view = np.ascontiguousarray(src.astype(np.float16, copy=False)).reshape(-1)
    total = batch * c1 * height * width_stride * c2
    if src_view.size < total:
      src_view = np.pad(src_view, (0, total - src_view.size))

    for n in range(batch):
      for c in range(channels):
        plane = c // c2
        offset = c % c2
        src_plane_base = ((n * c1 + plane) * plane_stride)
        for h in range(height):
          src_row_base = src_plane_base + h * width_stride * c2
          for w in range(width):
            src_idx = src_row_base + w * c2 + offset
            dst[n, c, h, w] = float(src_view[src_idx])
    return dst

  def _pack_nc1hwc2_fp16(self, src: np.ndarray, batch:int, channels:int, height:int, width:int,
                         c2:int, width_stride:int) -> np.ndarray:
    """
    Pack an NCHW tensor into the NC1HWC2 layout consumed by the Rockchip
    runtime. This is the inverse of `_unpack_nc1hwc2_fp16` and matches the
    DMA payload observed in `npu/ops_rknn/dump/gem2-dump`.
    """
    if batch <= 0 or channels <= 0 or height <= 0 or width <= 0 or width_stride <= 0 or c2 <= 0:
      return np.zeros(0, dtype=np.float16)
    c1 = max(1, math.ceil(channels / c2))
    plane_stride = height * width_stride * c2
    dst = np.zeros(batch * c1 * plane_stride, dtype=np.float16)
    src_view = np.ascontiguousarray(src.astype(np.float16, copy=False)).reshape(batch, channels, height, width)
    idx = 0
    for n in range(batch):
      for g in range(c1):
        for h in range(height):
          for w in range(width_stride):
            for c_slot in range(c2):
              channel = g * c2 + c_slot
              if channel < channels and w < width:
                dst[idx] = src_view[n, channel, h, w]
              else:
                dst[idx] = np.float16(0.0)
              idx += 1
    return dst

  def _pack_conv_weights_fp16(self, src: np.ndarray, out_channels:int, in_channels:int,
                              kernel_h:int, kernel_w:int, c2:int, c2_out:int) -> np.ndarray:
    """
    Arrange weights in the padded OIHW layout captured in the GEM2 dump. Each
    output kernel occupies `kernel_h * kernel_w * c2_out` scalars where the
    input channel axis is padded to `c2_out`.
    """
    if out_channels <= 0 or in_channels <= 0 or kernel_h <= 0 or kernel_w <= 0 or c2_out <= 0:
      return np.zeros(0, dtype=np.float16)
    kernel_stride = kernel_h * kernel_w * c2_out
    dst = np.zeros(out_channels * kernel_stride, dtype=np.float16)
    src_view = np.ascontiguousarray(src.astype(np.float16, copy=False)).reshape(out_channels, in_channels, kernel_h, kernel_w)
    for oc in range(out_channels):
      base_kernel = oc * kernel_stride
      for kh in range(kernel_h):
        for kw in range(kernel_w):
          dst_spatial_base = base_kernel + (kh * kernel_w + kw) * c2_out
          for ic in range(in_channels):
            dst[dst_spatial_base + ic] = src_view[oc, ic, kh, kw]
    return dst

  def _conv1d_hw(self, lhs_vec: np.ndarray, rhs_vec: np.ndarray, dtype: DType, np_dtype: np.dtype, out_len: int) -> np.ndarray|None:
    rhs_fp16 = rhs_vec.astype(np.float16, copy=False if np_dtype == np.float16 else True)
    rhs_bytes = rhs_fp16.tobytes()
    window_bytes = max(rhs_fp16.nbytes, 2)
    output_hw_bytes = max(4, np.dtype(np_dtype).itemsize)

    input_hw = self.device._gpu_alloc(window_bytes, 0)
    weight_hw = self.device._gpu_alloc(max(len(rhs_bytes), 2), 0)
    output_hw = self.device._gpu_alloc(output_hw_bytes, 0)

    ctypes.memmove(weight_hw.va_addr, rhs_bytes, len(rhs_bytes))

    result = np.empty(out_len, dtype=np.float32)
    regs_snapshot: list[int] | None = None
    input_width = rhs_fp16.size
    kernel_width = rhs_fp16.size
    output_width = 1
    in_channels = 1
    out_channels = 1
    channel_align = max(8, ((in_channels + 7) // 8) * 8)
    out_channel_align = max(16, ((out_channels + 15) // 16) * 16)
    input_width_aligned = (input_width + 15) & ~15
    dst_stride = output_width
    surface_add = dst_stride
    padded_kernel_bytes = kernel_width * channel_align * np.dtype(np.float16).itemsize

    for idx in range(out_len):
      window = lhs_vec[idx:idx+rhs_fp16.size]
      if window.size != rhs_fp16.size:
        raise RuntimeError("RK_CONV hardware window size mismatch")
      window_fp16 = window.astype(np.float16, copy=False if np_dtype == np.float16 else True)
      window_bytes_cur = window_fp16.tobytes()
      ctypes.memmove(input_hw.va_addr, window_bytes_cur, len(window_bytes_cur))
      ctypes.memset(output_hw.va_addr, 0, output_hw_bytes)

      self._program_conv1d_fp16(input_hw.meta.dma_addr, weight_hw.meta.dma_addr, output_hw.meta.dma_addr,
                                input_width, kernel_width, output_width,
                                in_channels, out_channels,
                                input_width_aligned, channel_align,
                                out_channel_align, dst_stride,
                                surface_add, padded_kernel_bytes,
                                batch_count=1)
      if regs_snapshot is None:
        regs_snapshot = list(self.q)
        if DEBUG:
          print("RK_CONV DMA", hex(input_hw.meta.dma_addr), hex(weight_hw.meta.dma_addr), hex(output_hw.meta.dma_addr))
      self._submit_conv()

      raw_output = ctypes.string_at(output_hw.va_addr, output_hw_bytes)
      if DEBUG and idx == 0:
        hex_len = 64 if DEBUG >= 3 else 32
        print("RK_CONV raw", raw_output[:hex_len].hex())
        print("RK_CONV regs len", len(regs_snapshot) if regs_snapshot else 0)
        if regs_snapshot:
          if DEBUG >= 3:
            print("RK_CONV regs full", [hex(v) for v in regs_snapshot])
          else:
            print("RK_CONV regs", [hex(v) for v in regs_snapshot[:10]])

      if output_hw_bytes >= 4:
        word = np.frombuffer(raw_output, dtype=np.uint32, count=1)[0]
        if (word >> 16) == 0:
          value = np.array([(word & 0xffff)], dtype=np.uint16).view(np.float16).astype(np.float32)[0]
        else:
          value = np.frombuffer(raw_output, dtype=np.float32, count=1)[0]
      else:
        value = np.frombuffer(raw_output, dtype=np.float16, count=1).astype(np.float32)[0]
      result[idx] = value

    return result

  def _conv2d_hw(self, lhs_arr: np.ndarray, rhs_arr: np.ndarray, dtype: DType, np_dtype: np.dtype) -> np.ndarray|None:
    """Evaluate a 3x3 NHWC convolution using the conv1d hardware helper.

    The layout assumptions mirror the RKNN sample in
    `/home/orangepi/tinygrad/npu/ops_rknn/conv2d_simple.cpp`; the register
    values come from the captured trace in
    `/home/orangepi/tinygrad/npu/ops_rknn/dump/conv2d.h`. Rather than replaying
    that entire sequence verbatim, each spatial patch is flattened and routed
    through `_conv1d_hw`, which already programs the Rockchip NPU with the same
    FP16 configuration observed in the trace.
    """

    supported_dtypes = {dtypes.float, dtypes.float32, dtypes.float16}
    if dtype not in supported_dtypes: return None
    if np_dtype not in (np.float16, np.float32): return None
    if lhs_arr.ndim != 4 or rhs_arr.ndim != 4: return None

    N, C, H, W = lhs_arr.shape
    K, Cw, KH, KW = rhs_arr.shape
    if C != Cw or C != 1 or N != 1: return None
    if KH != 3 or KW != 3: return None

    out_h = H - KH + 1
    out_w = W - KW + 1
    if out_h <= 0 or out_w <= 0: return None

    acc = np.zeros((N, K, out_h, out_w), dtype=np.float32)
    weight_cache: list[np.ndarray] = []
    for k in range(K):
      flat = rhs_arr[k].reshape(-1)
      weight_cache.append(np.ascontiguousarray(flat.astype(np_dtype, copy=False)))

    try:
      for n in range(N):
        for k in range(K):
          weight_vec = weight_cache[k]
          for y in range(out_h):
            for x in range(out_w):
              patch = lhs_arr[n, :, y:y+KH, x:x+KW]
              patch_vec = np.ascontiguousarray(patch.reshape(-1).astype(np_dtype, copy=False))
              hw_val = self._conv1d_hw(patch_vec, weight_vec, dtype, np_dtype, 1)
              if hw_val is None:
                return None
              acc[n, k, y, x] = float(hw_val[0])
    except Exception:
      if DEBUG:
        import traceback
        traceback.print_exc()
      return None

    return acc

  def _program_conv1d_fp16(self, input_dma: int, weight_dma: int, output_dma: int,
                           input_width: int, kernel_width: int, output_width: int,
                           in_channels: int, out_channels: int,
                           input_width_aligned: int, data_in_channel: int,
                           out_channel_align: int, dst_stride: int,
                           surface_add: int, padded_kernel_bytes: int,
                           batch_count: int, reset_queue: bool=True) -> None:
    feature_grains = 2
    data_in_height = 1
    weight_height = 1
    dataout_atomics = output_width
    weight_kernels = out_channels
    weight_bytes_per_kernel = padded_kernel_bytes
    weight_bytes_total = padded_kernel_bytes * out_channels
    out_channel_field = out_channel_align - 1
    data_cube_width = max(output_width - 1, 0)
    output_height_minus1 = 0
    orig_channel = max(out_channels - 1, 0)
    bytes_per_element = np.dtype(np.float16).itemsize
    row_bytes = dst_stride * out_channel_align * bytes_per_element

    self.q = []
    reg = self.reg
    emit = self.emit_raw

    emit(rk.CNA, rk.REG_CNA_CBUF_CON0,
      reg(11, rk.CNA_CBUF_CON0_WEIGHT_BANK__SHIFT, rk.CNA_CBUF_CON0_WEIGHT_BANK__MASK) |
      reg(1, rk.CNA_CBUF_CON0_DATA_BANK__SHIFT, rk.CNA_CBUF_CON0_DATA_BANK__MASK))
    emit(rk.CNA, rk.REG_CNA_DCOMP_REGNUM, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_CTRL, 0)
    conv_con1_val = (
      reg(1, rk.CNA_CONV_CON1_NONALIGN_DMA__SHIFT, rk.CNA_CONV_CON1_NONALIGN_DMA__MASK) |
      reg(1, rk.CNA_CONV_CON1_GROUP_LINE_OFF__SHIFT, rk.CNA_CONV_CON1_GROUP_LINE_OFF__MASK) |
      reg(8, rk.CNA_CONV_CON1_ARGB_IN__SHIFT, rk.CNA_CONV_CON1_ARGB_IN__MASK) |
      reg(2, rk.CNA_CONV_CON1_PROC_PRECISION__SHIFT, rk.CNA_CONV_CON1_PROC_PRECISION__MASK) |
      reg(2, rk.CNA_CONV_CON1_IN_PRECISION__SHIFT, rk.CNA_CONV_CON1_IN_PRECISION__MASK))
    emit(rk.CNA, rk.REG_CNA_CONV_CON1, conv_con1_val)
    emit(rk.DPU, rk.REG_DPU_S_POINTER,
      reg(1, rk.DPU_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_S_POINTER_POINTER_PP_MODE__MASK) |
      reg(1, rk.DPU_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_EN__MASK) |
      reg(1, rk.DPU_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_S_POINTER_POINTER_PP_EN__MASK))
    emit(rk.CNA, rk.REG_CNA_CONV_CON1, conv_con1_val)
    emit(rk.CNA, rk.REG_CNA_CONV_CON2,
      reg(feature_grains, rk.CNA_CONV_CON2_FEATURE_GRAINS__SHIFT, rk.CNA_CONV_CON2_FEATURE_GRAINS__MASK))
    emit(rk.CNA, rk.REG_CNA_CONV_CON3,
      reg(1, rk.CNA_CONV_CON3_CONV_Y_STRIDE__SHIFT, rk.CNA_CONV_CON3_CONV_Y_STRIDE__MASK) |
      reg(1, rk.CNA_CONV_CON3_CONV_X_STRIDE__SHIFT, rk.CNA_CONV_CON3_CONV_X_STRIDE__MASK))
    data_size0_val = (
      reg(input_width_aligned, rk.CNA_DATA_SIZE0_DATAIN_WIDTH__SHIFT, rk.CNA_DATA_SIZE0_DATAIN_WIDTH__MASK) |
      reg(data_in_height, rk.CNA_DATA_SIZE0_DATAIN_HEIGHT__SHIFT, rk.CNA_DATA_SIZE0_DATAIN_HEIGHT__MASK))
    emit(rk.CNA, rk.REG_CNA_DATA_SIZE0, data_size0_val)
    if DEBUG >= 2:
      print("RK_CONV reg CNA_DATA_SIZE0 width", input_width_aligned,
            "height", data_in_height, "word", hex(data_size0_val))
    emit(rk.CNA, rk.REG_CNA_DATA_SIZE1,
      reg(data_in_channel, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL__SHIFT, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL__MASK))
    emit(rk.CNA, rk.REG_CNA_DATA_SIZE2,
      reg(output_width, rk.CNA_DATA_SIZE2_DATAOUT_WIDTH__SHIFT, rk.CNA_DATA_SIZE2_DATAOUT_WIDTH__MASK))
    data_size3_val = reg(dataout_atomics, rk.CNA_DATA_SIZE3_DATAOUT_ATOMICS__SHIFT, rk.CNA_DATA_SIZE3_DATAOUT_ATOMICS__MASK)
    emit(rk.CNA, rk.REG_CNA_DATA_SIZE3, data_size3_val)
    if DEBUG >= 2:
      print("RK_CONV reg CNA_DATA_SIZE3 atomics", dataout_atomics, "word", hex(data_size3_val))
    emit(rk.CNA, rk.REG_CNA_WEIGHT_SIZE0, weight_bytes_total)
    emit(rk.CNA, rk.REG_CNA_WEIGHT_SIZE1,
      reg(weight_bytes_per_kernel, rk.CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL__SHIFT, rk.CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL__MASK))
    emit(rk.CNA, rk.REG_CNA_WEIGHT_SIZE2,
      reg(kernel_width, rk.CNA_WEIGHT_SIZE2_WEIGHT_WIDTH__SHIFT, rk.CNA_WEIGHT_SIZE2_WEIGHT_WIDTH__MASK) |
      reg(weight_height, rk.CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT__SHIFT, rk.CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT__MASK) |
      reg(weight_kernels, rk.CNA_WEIGHT_SIZE2_WEIGHT_KERNELS__SHIFT, rk.CNA_WEIGHT_SIZE2_WEIGHT_KERNELS__MASK))
    emit(rk.CNA, rk.REG_CNA_CBUF_CON0,
      reg(11, rk.CNA_CBUF_CON0_WEIGHT_BANK__SHIFT, rk.CNA_CBUF_CON0_WEIGHT_BANK__MASK) |
      reg(1, rk.CNA_CBUF_CON0_DATA_BANK__SHIFT, rk.CNA_CBUF_CON0_DATA_BANK__MASK))
    emit(rk.CNA, rk.REG_CNA_CBUF_CON1,
      reg(input_width_aligned, rk.CNA_CBUF_CON1_DATA_ENTRIES__SHIFT, rk.CNA_CBUF_CON1_DATA_ENTRIES__MASK))
    emit(rk.CNA, rk.REG_CNA_CVT_CON0,
      reg(1, rk.CNA_CVT_CON0_CVT_BYPASS__SHIFT, rk.CNA_CVT_CON0_CVT_BYPASS__MASK))
    emit(rk.CNA, rk.REG_CNA_CVT_CON1,
      reg(1, rk.CNA_CVT_CON1_CVT_SCALE0__SHIFT, rk.CNA_CVT_CON1_CVT_SCALE0__MASK))
    emit(rk.CNA, rk.REG_CNA_CVT_CON2,
      reg(1, rk.CNA_CVT_CON2_CVT_SCALE1__SHIFT, rk.CNA_CVT_CON2_CVT_SCALE1__MASK))
    emit(rk.CNA, rk.REG_CNA_CVT_CON3,
      reg(1, rk.CNA_CVT_CON3_CVT_SCALE2__SHIFT, rk.CNA_CVT_CON3_CVT_SCALE2__MASK))
    emit(rk.CNA, rk.REG_CNA_CVT_CON4,
      reg(1, rk.CNA_CVT_CON4_CVT_SCALE3__SHIFT, rk.CNA_CVT_CON4_CVT_SCALE3__MASK))
    emit(rk.CNA, rk.REG_CNA_FC_CON0, 0)
    emit(rk.CNA, rk.REG_CNA_FC_CON1, 0)
    emit(rk.CNA, rk.REG_CNA_PAD_CON0, 0)
    emit(rk.CNA, rk.REG_CNA_FEATURE_DATA_ADDR,
      reg(input_dma, rk.CNA_FEATURE_DATA_ADDR_FEATURE_BASE_ADDR__SHIFT, rk.CNA_FEATURE_DATA_ADDR_FEATURE_BASE_ADDR__MASK))
    emit(rk.CNA, rk.REG_CNA_FC_CON2, 0)
    emit(rk.CNA, rk.REG_CNA_DMA_CON0,
      reg(15, rk.CNA_DMA_CON0_WEIGHT_BURST_LEN__SHIFT, rk.CNA_DMA_CON0_WEIGHT_BURST_LEN__MASK) |
      reg(15, rk.CNA_DMA_CON0_DATA_BURST_LEN__SHIFT, rk.CNA_DMA_CON0_DATA_BURST_LEN__MASK))
    emit(rk.CNA, rk.REG_CNA_DMA_CON1,
      reg(input_width_aligned, rk.CNA_DMA_CON1_LINE_STRIDE__SHIFT, rk.CNA_DMA_CON1_LINE_STRIDE__MASK))
    emit(rk.CNA, rk.REG_CNA_DMA_CON2, 0)
    emit(rk.CNA, rk.REG_CNA_FC_DATA_SIZE0,
      reg(input_width, rk.CNA_FC_DATA_SIZE0_DMA_WIDTH__SHIFT, rk.CNA_FC_DATA_SIZE0_DMA_WIDTH__MASK) |
      reg(data_in_height, rk.CNA_FC_DATA_SIZE0_DMA_HEIGHT__SHIFT, rk.CNA_FC_DATA_SIZE0_DMA_HEIGHT__MASK))
    emit(rk.CNA, rk.REG_CNA_FC_DATA_SIZE1,
      reg(data_in_channel, rk.CNA_FC_DATA_SIZE1_DMA_CHANNEL__SHIFT, rk.CNA_FC_DATA_SIZE1_DMA_CHANNEL__MASK))
    emit(rk.CNA, rk.REG_CNA_DCOMP_CTRL, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_REGNUM, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_ADDR0,
      reg(weight_dma, rk.CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0__SHIFT, rk.CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0__MASK))
    for offset in range(16):
      reg_name = f"REG_CNA_DCOMP_AMOUNT{offset}"
      if hasattr(rk, reg_name):
        emit(rk.CNA, getattr(rk, reg_name), 0)
    emit(rk.CNA, rk.REG_CNA_CVT_CON5, 0x0000ffff)
    emit(rk.CNA, rk.REG_CNA_PAD_CON1, 0)
    emit(rk.CORE, rk.REG_CORE_MISC_CFG,
      reg(2, rk.CORE_MISC_CFG_PROC_PRECISION__SHIFT, rk.CORE_MISC_CFG_PROC_PRECISION__MASK))
    core_dataout_size_0_val = (
      reg(output_height_minus1, rk.CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT__SHIFT, rk.CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT__MASK) |
      reg(data_cube_width, rk.CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH__SHIFT, rk.CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH__MASK))
    emit(rk.CORE, rk.REG_CORE_DATAOUT_SIZE_0, core_dataout_size_0_val)
    if DEBUG >= 2:
      print("RK_CONV reg CORE_DATAOUT_SIZE_0 height", output_height_minus1,
            "width", data_cube_width, "word", hex(core_dataout_size_0_val))
    core_dataout_size_1_val = reg(out_channel_field, rk.CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL__SHIFT, rk.CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL__MASK)
    emit(rk.CORE, rk.REG_CORE_DATAOUT_SIZE_1, core_dataout_size_1_val)
    if DEBUG >= 2:
      print("RK_CONV reg CORE_DATAOUT_SIZE_1 channels", out_channel_field, "word", hex(core_dataout_size_1_val))
    emit(rk.CORE, rk.REG_CORE_CLIP_TRUNCATE, 0)
    self.emit_raw(rk.CORE, 0x3030, 0)
    emit(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
      reg(15, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(2, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_FORMAT,
      reg(2, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
      reg(2, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
      reg(2, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))
    emit(rk.DPU, rk.REG_DPU_OFFSET_PEND, 0)
    emit(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
      reg(output_dma, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
    dpu_dst_surf_stride_val = reg(dst_stride, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK)
    emit(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE, dpu_dst_surf_stride_val)
    if DEBUG >= 2:
      print("RK_CONV reg DPU_DST_SURF_STRIDE dst_stride", dst_stride, "word", hex(dpu_dst_surf_stride_val))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
    dpu_data_cube_height_val = reg(output_height_minus1, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__MASK)
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT, dpu_data_cube_height_val)
    if DEBUG >= 2:
      print("RK_CONV reg DPU_DATA_CUBE_HEIGHT height_minus1", output_height_minus1, "word", hex(dpu_data_cube_height_val))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, 0)
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
      reg(orig_channel, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__MASK) |
      reg(out_channel_field, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_CFG,
      reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_MUL_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_MUL_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_ALU_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_ALU_CFG, 0)
    emit(rk.DPU, rk.REG_DPU_BS_MUL_CFG, 0)
    emit(rk.DPU, rk.REG_DPU_BS_RELUX_CMP_VALUE, 0)
    emit(rk.DPU, rk.REG_DPU_BS_OW_CFG,
      reg(1, rk.DPU_BS_OW_CFG_SIZE_E_2__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_2__MASK) |
      reg(1, rk.DPU_BS_OW_CFG_SIZE_E_1__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_1__MASK) |
      reg(1, rk.DPU_BS_OW_CFG_SIZE_E_0__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_0__MASK) |
      reg(1, rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT, rk.DPU_BS_OW_CFG_OD_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_OW_OP, 0)
    dpu_wdma_size_0_val = reg(out_channel_field, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__SHIFT, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK)
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_0, dpu_wdma_size_0_val)
    if DEBUG >= 2:
      print("RK_CONV reg DPU_WDMA_SIZE_0 channel_field", out_channel_field, "word", hex(dpu_wdma_size_0_val))
    dpu_wdma_size_1_val = (
      reg(output_height_minus1, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__MASK) |
      reg(data_cube_width, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_1, dpu_wdma_size_1_val)
    if DEBUG >= 2:
      print("RK_CONV reg DPU_WDMA_SIZE_1 height", output_height_minus1,
            "width", data_cube_width, "word", hex(dpu_wdma_size_1_val))
    emit(rk.DPU, rk.REG_DPU_BN_CFG,
      reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_MUL_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_BN_ALU_CFG, 0)
    emit(rk.DPU, rk.REG_DPU_BN_MUL_CFG, 0)
    emit(rk.DPU, rk.REG_DPU_BN_RELUX_CMP_VALUE, 0)
    emit(rk.DPU, rk.REG_DPU_EW_CFG,
      reg(1, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_EW_CVT_OFFSET_VALUE, 0)
    emit(rk.DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE,
      reg(1, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__SHIFT, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_EW_RELUX_CMP_VALUE, 0)
    emit(rk.DPU, rk.REG_DPU_OUT_CVT_OFFSET, 0)
    emit(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE,
      reg(1, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__SHIFT, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__MASK) |
      reg(1, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__SHIFT, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT, 0)
    emit(rk.DPU, rk.REG_DPU_EW_OP_VALUE_0, 0)
    emit(rk.DPU, rk.REG_DPU_EW_OP_VALUE_1, 0)
    emit(rk.DPU, rk.REG_DPU_EW_OP_VALUE_2, 0)
    emit(rk.DPU, rk.REG_DPU_EW_OP_VALUE_3, 0)
    emit(rk.DPU, rk.REG_DPU_EW_OP_VALUE_4, 0)
    emit(rk.DPU, rk.REG_DPU_EW_OP_VALUE_5, 0)
    emit(rk.DPU, rk.REG_DPU_EW_OP_VALUE_6, 0)
    emit(rk.DPU, rk.REG_DPU_EW_OP_VALUE_7, 0)
    dpu_surface_add_val = reg(surface_add, rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT, rk.DPU_SURFACE_ADD_SURF_ADD__MASK)
    emit(rk.DPU, rk.REG_DPU_SURFACE_ADD, dpu_surface_add_val)
    if DEBUG >= 2:
      print("RK_CONV reg DPU_SURFACE_ADD surface_add", surface_add,
            "row_bytes", row_bytes, "word", hex(dpu_surface_add_val))
    emit(rk.DPU, rk.REG_DPU_LUT_ACCESS_CFG, 0)
    emit(rk.DPU, rk.REG_DPU_LUT_ACCESS_DATA, 0)
    emit(rk.DPU, rk.REG_DPU_LUT_CFG, 0)
    emit(rk.DPU, rk.REG_DPU_LUT_INFO, 0)
    emit(rk.DPU, rk.REG_DPU_LUT_LE_START, 0)
    emit(rk.DPU, rk.REG_DPU_LUT_LE_END, 0)
    emit(rk.DPU, rk.REG_DPU_LUT_LO_START, 0)
    emit(rk.DPU, rk.REG_DPU_LUT_LO_END, 0)
    emit(rk.DPU, rk.REG_DPU_LUT_LE_SLOPE_SCALE, 0)
    emit(rk.DPU, rk.REG_DPU_LUT_LE_SLOPE_SHIFT, 0)
    emit(rk.DPU, rk.REG_DPU_LUT_LO_SLOPE_SCALE, 0)
    emit(rk.DPU, rk.REG_DPU_LUT_LO_SLOPE_SHIFT, 0)
    emit(rk.DPU, rk.REG_PC_REGISTER_AMOUNTS, 0)
    emit(rk.DPU, rk.REG_PC_VERSION, 0)
    self.emit_raw(0x0, 0x40c4, 0)
    self.emit_raw(0x80, rk.REG_PC_OPERATION_ENABLE,
      reg(6, rk.PC_OPERATION_ENABLE_RESERVED_0__SHIFT, rk.PC_OPERATION_ENABLE_RESERVED_0__MASK) |
      reg(1, rk.PC_OPERATION_ENABLE_OP_EN__SHIFT, rk.PC_OPERATION_ENABLE_OP_EN__MASK))
    if reset_queue:
      self._rk_conv_debug = {
        "dma": (input_dma, weight_dma, output_dma),
        "dst_stride": dst_stride,
        "surface_add": surface_add,
        "batch_count": batch_count,
        "row_bytes": row_bytes,
        "out_channel_align": out_channel_align,
        "data_cube_width": data_cube_width,
        "output_height_minus1": output_height_minus1,
        "dataout_atomics": dataout_atomics,
      }
      if DEBUG >= 2:
        print("RK_CONV register words", len(self.q))

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
    lhs_shape_flat = info.lhs_tensor_shape or info.lhs_base_shape or info.lhs_shape
    rhs_shape_flat = info.rhs_tensor_shape or info.rhs_base_shape or info.rhs_shape
    out_shape_write = info.out_tensor_shape or info.out_base_shape or info.out_shape
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
      print("lhs shape", lhs_shape_flat, "rhs shape", rhs_shape_flat, "out shape", out_shape_write)
    np_dtype = _np_dtype(dtype)
    post_ops = info.post_ops

    def _shape_prod(shape: tuple[Any, ...]|None) -> int:
      if not shape: return 0
      prod = 1
      for dim in shape:
        prod *= int(dim)
      return prod

    lhs_elems = _shape_prod(lhs_shape_flat)
    rhs_elems = _shape_prod(rhs_shape_flat)
    out_elems = _shape_prod(out_shape_write)
    axes = info.axes or tuple()

    conv2d_dims: tuple[int, int, int, int, int, int, int, int, int]|None = None
    if (lhs_shape_flat and rhs_shape_flat and out_shape_write and len(lhs_shape_flat) == 1
        and len(rhs_shape_flat) == 1 and len(axes) == 2 and info.lhs_shape):
      try:
        KH = int(info.lhs_shape[axes[0]])
        KW = int(info.lhs_shape[axes[1]])
      except Exception:
        KH = KW = None
      if KH and KW and rhs_elems == KH * KW and lhs_elems and out_elems:
        for outH in range(1, out_elems + 1):
          if out_elems % outH != 0:
            continue
          outW = out_elems // outH
          H = outH + KH - 1
          W = outW + KW - 1
          if H * W == lhs_elems:
            conv2d_dims = (1, 1, H, W, 1, KH, KW, outH, outW)
            break

    if conv2d_dims is not None:
      N, C, H, W, K, KH, KW, outH, outW = conv2d_dims
      lhs_arr = np.frombuffer(self._buffer_as_bytes(lhs_buf), dtype=np_dtype, count=lhs_elems).reshape((N, C, H, W))
      rhs_arr = np.frombuffer(self._buffer_as_bytes(rhs_buf), dtype=np_dtype, count=rhs_elems).reshape((K, C, KH, KW))
      try:
        hw_arr = self._conv2d_hw(lhs_arr, rhs_arr, dtype, np_dtype)
      except Exception as exc:
        if DEBUG:
          import traceback
          print("RK_CONV conv2d_hw failed", exc)
          traceback.print_exc()
        raise RuntimeError("RK_CONV conv2d hardware path failed") from exc
      if hw_arr is None:
        raise RuntimeError("RK_CONV conv2d hardware path returned no result")
      if post_ops:
        hw_arr = self._apply_post_ops_array(hw_arr, post_ops)
      if hw_arr.dtype != np_dtype:
        hw_arr = hw_arr.astype(np_dtype)
      out_view = hw_arr.reshape(tuple(int(x) for x in out_shape_write))
      self._write_bytes(out_buf, out_view.tobytes())
      return 0.0

    if DEBUG >= 2:
      print("RK_CONV shapes", lhs_shape_flat, rhs_shape_flat)
    if (lhs_shape_flat and rhs_shape_flat and len(lhs_shape_flat) <= 3
        and len(rhs_shape_flat) <= 3):
      if DEBUG >= 2:
        print("RK_CONV 1d candidate", lhs_shape_flat, rhs_shape_flat)
      lhs_len_single = int(lhs_elems)
      rhs_len_single = int(rhs_elems)
      lhs_vec = np.frombuffer(self._buffer_as_bytes(lhs_buf), dtype=np_dtype)
      rhs_vec = np.frombuffer(self._buffer_as_bytes(rhs_buf), dtype=np_dtype)

      output_width = lhs_len_single - rhs_len_single + 1
      if output_width <= 0:
        raise RuntimeError("invalid 1D convolution dimensions")
      dims = None
      per_sample_elems = batch_count = output_elems_per_sample = total_output_elements = 0
      error: Exception|None = None
      result_arr: np.ndarray|None = None
      try:
        dims = self._conv1d_shape_info(info, lhs_len_single, rhs_len_single)
        if DEBUG >= 2:
          print("RK_CONV 1d dims", dims)
        if dims is None:
          raise RuntimeError("RK_CONV metadata does not describe a valid RKNC1d")
        input_width, kernel_width, output_width, in_channels, out_channels, groups, stride, dilation = dims
        per_sample_elems = input_width * in_channels
        if per_sample_elems <= 0 or lhs_len_single < per_sample_elems:
          raise RuntimeError("invalid 1D convolution buffer size for dims %s" % (dims,))
        if lhs_len_single % per_sample_elems != 0:
          raise RuntimeError("RK_CONV 1d buffer length not divisible by sample size %s" % (dims,))
        batch_count = lhs_len_single // per_sample_elems
        if DEBUG >= 2 and batch_count > 1:
          print("RK_CONV 1d batches", batch_count)
        output_elems_per_sample = out_channels * output_width
        total_output_elements = batch_count * output_elems_per_sample
        result_arr = self._conv1d_hw_full(lhs_vec, rhs_vec, dtype, np_dtype, *dims, batch_count=batch_count)
        if result_arr is None:
          raise RuntimeError("RK_CONV hardware path returned no data for dims %s" % (dims,))
        if result_arr.size != total_output_elements:
          raise RuntimeError("RK_CONV output size mismatch for dims %s" % (dims,))
      except Exception as exc:
        error = exc
        if DEBUG:
          print("RK_CONV multi-batch path failed, falling back:", exc)
          import traceback
          traceback.print_exc()
        result_arr = None
      if result_arr is None:
        if dims is None:
          raise RuntimeError("RK_CONV metadata does not describe a valid RKNC1d") from (error or None)
        if DEBUG >= 2:
          print("RK_CONV fallback splits", batch_count, "batches")
        batch_results = np.empty((batch_count, output_elems_per_sample), dtype=np_dtype)
        for batch_idx in range(batch_count):
          offset = batch_idx * per_sample_elems
          sample = lhs_vec[offset:offset + per_sample_elems]
          if sample.size != per_sample_elems:
            raise RuntimeError("RK_CONV sample size mismatch for dims %s" % (dims,))
          batch_arr = self._conv1d_hw_full(sample, rhs_vec, dtype, np_dtype, *dims, batch_count=1)
          if batch_arr is None:
            raise RuntimeError("RK_CONV hardware fallback failed for dims %s on batch %d" % (dims, batch_idx))
          if batch_arr.size != output_elems_per_sample:
            raise RuntimeError("RK_CONV output size mismatch for dims %s" % (dims,))
          batch_results[batch_idx] = batch_arr
        result_arr = batch_results.reshape(total_output_elements)

      if post_ops:
        result_arr = self._apply_post_ops_array(result_arr, post_ops)
      reshaped = result_arr.reshape(tuple(int(x) for x in out_shape_write))
      self._write_bytes(out_buf, reshaped.tobytes())
      return 0.0

    lhs_shape = lhs_shape_flat
    rhs_shape = rhs_shape_flat
    out_shape = out_shape_write
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

    hw_arr = None
    try:
      hw_arr = self._conv2d_hw(lhs_arr, rhs_arr, dtype, np_dtype)
    except Exception as exc:
      if DEBUG:
        import traceback
        print("RK_CONV conv2d_hw failed", exc)
        traceback.print_exc()
      raise RuntimeError("RK_CONV conv2d hardware path failed") from exc
    if hw_arr is None:
      raise RuntimeError("RK_CONV conv2d hardware path returned no result")
    if post_ops:
      hw_arr = self._apply_post_ops_array(hw_arr, post_ops)
    if hw_arr.dtype != np_dtype:
      hw_arr = hw_arr.astype(np_dtype)
    hw_view = hw_arr.reshape(out_shape)
    self._write_bytes(out_buf, hw_view.tobytes())
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
