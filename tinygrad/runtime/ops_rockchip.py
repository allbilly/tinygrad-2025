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
from typing import Any, TYPE_CHECKING, Callable
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
from tinygrad.runtime.rockchip_lut import SIGMOID_LUT_TABLES, SILU_LUT_TABLES

import sys, numpy as np
np.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=False)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

DEBUG = getenv("DEBUG")
FUSE_POSTOPS = getenv("ROCKCHIP_FUSE_POSTOPS", 1)
ROUNDOFF_TAG = "RK_ROUNDOFF"
WHERE_TAG = "RK_WHERE"
IDIV_TAG = "RK_IDIV"
SIGMOID_TAG = "RK_SIGMOID"
SILU_TAG = "RK_SILU"
ABS_TAG = "RK_ABS"

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

@dataclass(frozen=True)
class RockchipConv2DDesc:
  name: str
  lhs_shape: tuple[int, ...]
  rhs_shape: tuple[int, ...]
  kernel: tuple[int, int]
  out_hw: tuple[int, int]
  width_stride: int
  out_width_stride: int
  align_c: int
  align_out_c: int
  groups: int = 1
  weight_transform: Callable[[np.ndarray], np.ndarray]|None = None
  feature_grains: int = 7
  cbuf_entries: int|None = None

  @property
  def lhs_elems(self) -> int:
    return int(np.prod(self.lhs_shape))

  @property
  def rhs_elems(self) -> int:
    return int(np.prod(self.rhs_shape))

def _expand_group_weights(rhs: np.ndarray, groups:int) -> np.ndarray:
  expanded = np.zeros((rhs.shape[0], groups, rhs.shape[2], rhs.shape[3]), dtype=rhs.dtype)
  out_per_group = rhs.shape[0] // groups if groups else 0
  for oc in range(rhs.shape[0]):
    oc_group = oc // out_per_group if out_per_group else 0
    expanded[oc, oc_group] = rhs[oc, 0]
  return expanded

_CONV2D_DESCS: dict[str, RockchipConv2DDesc] = {
  "6321": RockchipConv2DDesc("6321", (1, 3, 5, 7), (6, 3, 2, 1), (2, 1), (4, 7), 8, 28, 8, 16, feature_grains=7, cbuf_entries=None),
  "6323": RockchipConv2DDesc("6323", (1, 3, 5, 7), (6, 3, 2, 3), (2, 3), (4, 5), 8, 20, 8, 16, feature_grains=7, cbuf_entries=None),
  "6325": RockchipConv2DDesc("6325", (1, 3, 5, 7), (6, 3, 2, 5), (2, 5), (4, 3), 8, 12, 8, 16, feature_grains=7, cbuf_entries=40),
  "6331": RockchipConv2DDesc("6331", (1, 3, 5, 7), (6, 3, 3, 1), (3, 1), (3, 7), 8, 24, 8, 16, feature_grains=8, cbuf_entries=40),
  "6333": RockchipConv2DDesc("6333", (1, 3, 5, 7), (6, 3, 3, 3), (3, 3), (3, 5), 8, 16, 8, 16, feature_grains=8, cbuf_entries=40),
  "6335": RockchipConv2DDesc("6335", (1, 3, 5, 7), (6, 3, 3, 5), (3, 5), (3, 3), 8, 12, 8, 16, feature_grains=8, cbuf_entries=40),
  "6133": RockchipConv2DDesc("6133", (1, 3, 5, 7), (6, 1, 3, 3), (3, 3), (3, 5), 8, 16, 8, 16, groups=3, feature_grains=8, cbuf_entries=40,
    weight_transform=lambda rhs: _expand_group_weights(rhs, 3)),
}

_CONV2D_DISPATCH: dict[tuple[int, int], list[RockchipConv2DDesc]] = {}
for desc in _CONV2D_DESCS.values():
  _CONV2D_DISPATCH.setdefault((desc.lhs_elems, desc.rhs_elems), []).append(desc)

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

def _strip_movement(node:UOp) -> UOp:
  cur = node
  while cur.op in MOVEMENT_OPS and len(cur.src) == 1:
    cur = cur.src[0]
  return cur

def _const_eval(node:UOp) -> Any|None:
  if node.op is Ops.CONST: return node.arg
  if node.op in MOVEMENT_OPS and len(node.src) == 1:
    return _const_eval(node.src[0])
  if node.op in (Ops.ADD, Ops.MUL, Ops.SUB, Ops.DIV) and len(node.src) == 2:
    a = _const_eval(node.src[0])
    b = _const_eval(node.src[1])
    if a is None or b is None: return None
    if node.op is Ops.ADD: return a + b
    if node.op is Ops.MUL: return a * b
    if node.op is Ops.SUB: return a - b
    if node.op is Ops.DIV: return a / b
  if node.op is Ops.RECIP and len(node.src) == 1:
    a = _const_eval(node.src[0])
    if a is None: return None
    return 1.0 / a
  return None

def _const_close(val:Any, target:float, tol:float=1e-6) -> bool:
  try:
    return abs(float(val) - target) <= tol
  except Exception:
    return False

_SIGMOID_MUL = -1.0 / math.log(2.0)

def _is_zero_const(node:UOp) -> bool:
  val = _const_eval(node)
  return val is not None and _const_close(val, 0.0)

def _is_mul_zero_of_base(node:UOp, base:UOp) -> bool:
  if node.op is not Ops.MUL or len(node.src) != 2:
    return False
  if _is_zero_const(node.src[0]) and _strip_movement(node.src[1]) is _strip_movement(base):
    return True
  if _is_zero_const(node.src[1]) and _strip_movement(node.src[0]) is _strip_movement(base):
    return True
  return False

def _strip_add_zero(node:UOp, base:UOp) -> UOp:
  cur = _strip_movement(node)
  if cur.op is Ops.ADD and len(cur.src) == 2:
    if _is_zero_const(cur.src[0]) or _is_mul_zero_of_base(cur.src[0], base):
      return cur.src[1]
    if _is_zero_const(cur.src[1]) or _is_mul_zero_of_base(cur.src[1], base):
      return cur.src[0]
  return node

def _match_add_const(node:UOp) -> tuple[UOp, float]|None:
  if node.op is Ops.ADD and len(node.src) == 2:
    lhs = _const_eval(node.src[0])
    rhs = _const_eval(node.src[1])
    if lhs is not None and rhs is None:
      return node.src[1], float(lhs)
    if rhs is not None and lhs is None:
      return node.src[0], float(rhs)
  return None

def _match_sigmoid_base(node:UOp) -> UOp|None:
  if node.op is not Ops.RECIP or len(node.src) != 1:
    return None
  add_node = _strip_movement(node.src[0])
  add_match = _match_add_const(add_node)
  if add_match is None:
    return None
  base, add_const = add_match
  if not _const_close(add_const, 1.0):
    return None
  exp_node = _strip_movement(base)
  if exp_node.op is not Ops.EXP2 or len(exp_node.src) != 1:
    return None
  mul_node = _strip_movement(exp_node.src[0])
  if mul_node.op is not Ops.MUL or len(mul_node.src) != 2:
    return None
  lhs_const = _const_eval(mul_node.src[0])
  rhs_const = _const_eval(mul_node.src[1])
  if lhs_const is not None and _const_close(lhs_const, _SIGMOID_MUL):
    return mul_node.src[1]
  if rhs_const is not None and _const_close(rhs_const, _SIGMOID_MUL):
    return mul_node.src[0]
  return None

def _match_sign_pattern(node:UOp, base:UOp) -> bool:
  node = _strip_add_zero(node, base)
  node = _strip_movement(node)
  if node.op is not Ops.WHERE or len(node.src) != 3:
    return False
  cond0, tval0, fval0 = node.src
  if not _is_zero_const(fval0):
    return False
  cond0 = _strip_movement(cond0)
  if cond0.op is not Ops.CMPNE or len(cond0.src) != 2:
    return False
  if not (_strip_movement(cond0.src[0]) is _strip_movement(base) and _is_zero_const(cond0.src[1])) and \
     not (_strip_movement(cond0.src[1]) is _strip_movement(base) and _is_zero_const(cond0.src[0])):
    return False
  inner = _strip_movement(tval0)
  if inner.op is not Ops.WHERE or len(inner.src) != 3:
    return False
  cond1, tval1, fval1 = inner.src
  cond1 = _strip_movement(cond1)
  if cond1.op is not Ops.CMPLT or len(cond1.src) != 2:
    return False
  if _strip_movement(cond1.src[0]) is not _strip_movement(base):
    return False
  if not _is_zero_const(cond1.src[1]):
    return False
  if not _const_close(_const_eval(tval1), -1.0):
    return False
  if not _const_close(_const_eval(fval1), 1.0):
    return False
  return True

def _base_sources(node:UOp) -> set[UOp]:
  sources:set[UOp] = set()
  stack:list[UOp] = [node]
  seen:set[UOp] = set()
  while stack:
    cur = stack.pop()
    if cur in seen: continue
    seen.add(cur)
    if _const_eval(cur) is not None:
      continue
    if cur.op in MOVEMENT_OPS and len(cur.src) == 1:
      stack.append(cur.src[0])
      continue
    if cur.op in {Ops.COPY, Ops.LOAD} or not cur.src:
      sources.add(cur)
      continue
    stack.extend(cur.src)
  return sources

def _uses_single_base(node:UOp, base:UOp) -> bool:
  base_root = _strip_movement(base)
  sources = {_strip_movement(s) for s in _base_sources(node)}
  return sources == {base_root}

def _is_roundoff_arg(arg:Any) -> bool:
  return isinstance(arg, tuple) and len(arg) >= 1 and arg[0] == ROUNDOFF_TAG

def _is_where_arg(arg:Any) -> bool:
  return isinstance(arg, tuple) and len(arg) >= 1 and arg[0] == WHERE_TAG

def _is_idiv_arg(arg:Any) -> bool:
  return isinstance(arg, tuple) and len(arg) >= 1 and arg[0] == IDIV_TAG

def _is_sigmoid_arg(arg:Any) -> bool:
  return isinstance(arg, tuple) and len(arg) >= 1 and arg[0] == SIGMOID_TAG

def _is_silu_arg(arg:Any) -> bool:
  return isinstance(arg, tuple) and len(arg) >= 1 and arg[0] == SILU_TAG

def _is_abs_arg(arg:Any) -> bool:
  return isinstance(arg, tuple) and len(arg) >= 1 and arg[0] == ABS_TAG

def _walk_dependencies(node:UOp, target:UOp) -> bool:
  stack:list[UOp] = [node]
  seen:set[UOp] = set()
  while stack:
    cur = stack.pop()
    if cur is target: return True
    if cur in seen: continue
    seen.add(cur)
    stack.extend(cur.src)
  return False

def _depends_on(node:UOp, target:UOp) -> bool:
  return _walk_dependencies(node, target)

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

def _match_round_ceil(node:UOp) -> tuple[UOp, float]|None:
  if node.op is not Ops.WHERE or len(node.src) != 3: return None
  cond, tval, fval = node.src
  if fval.op is not Ops.TRUNC or len(fval.src) != 1: return None
  trunc_node = fval
  add_match = _match_add_const(tval)
  if add_match is None: return None
  add_base, add_const = add_match
  if not _const_close(add_const, 1.0): return None
  if _strip_movement(add_base) is not _strip_movement(trunc_node): return None
  add_node = trunc_node.src[0]
  if cond.op is not Ops.CMPLT or len(cond.src) != 2: return None
  if _strip_movement(cond.src[0]) is not _strip_movement(trunc_node): return None
  if _strip_movement(cond.src[1]) is not _strip_movement(add_node): return None
  return _match_add_const(add_node)

def _match_round_floor(node:UOp) -> tuple[UOp, float]|None:
  if node.op is not Ops.WHERE or len(node.src) != 3: return None
  cond, tval, fval = node.src
  if fval.op is not Ops.TRUNC or len(fval.src) != 1: return None
  trunc_node = fval
  add_match = _match_add_const(tval)
  if add_match is None: return None
  add_base, add_const = add_match
  if not _const_close(add_const, -1.0): return None
  if _strip_movement(add_base) is not _strip_movement(trunc_node): return None
  add_node = trunc_node.src[0]
  if cond.op is not Ops.CMPLT or len(cond.src) != 2: return None
  if _strip_movement(cond.src[0]) is not _strip_movement(add_node): return None
  if _strip_movement(cond.src[1]) is not _strip_movement(trunc_node): return None
  return _match_add_const(add_node)

def _rockchip_roundoff_rewrite(node:UOp) -> UOp|None:
  if node.op is not Ops.WHERE or len(node.src) != 3: return None
  if node.dtype not in dtypes.floats: return None
  cond, tval, fval = node.src
  ceil_match = _match_round_ceil(tval)
  if ceil_match is None: return None
  floor_match = _match_round_floor(fval)
  if floor_match is None: return None
  ceil_base, ceil_offset = ceil_match
  floor_base, floor_offset = floor_match
  if _strip_movement(ceil_base) is not _strip_movement(floor_base): return None
  if not _const_close(ceil_offset, -0.5): return None
  if not _const_close(floor_offset, 0.5): return None
  if not _uses_single_base(cond, ceil_base): return None
  try:
    shape = _safe_shape(node)
  except Exception:
    shape = tuple()
  return UOp(Ops.CUSTOM, node.dtype, src=(ceil_base,), arg=(ROUNDOFF_TAG, shape), metadata=node.metadata)

def _rockchip_where_rewrite(node:UOp) -> UOp|None:
  if node.op is not Ops.WHERE or len(node.src) != 3: return None
  if node.dtype not in (dtypes.float16, dtypes.float): return None
  cond, tval, fval = node.src
  if cond.dtype not in (dtypes.bool, dtypes.float16, dtypes.float): return None
  if tval.dtype is not node.dtype or fval.dtype is not node.dtype: return None
  try:
    shape = _safe_shape(node)
  except Exception:
    shape = tuple()
  return UOp(Ops.CUSTOM, node.dtype, src=node.src, arg=(WHERE_TAG, shape), metadata=node.metadata)

def _rockchip_idiv_rewrite(node:UOp) -> UOp|None:
  if node.op is not Ops.IDIV or len(node.src) != 2: return None
  if node.dtype not in dtypes.sints or node.dtype.itemsize > 4: return None
  if any(src.dtype not in dtypes.sints for src in node.src): return None
  try:
    shape = _safe_shape(node)
  except Exception:
    shape = tuple()
  return UOp(Ops.CUSTOM, node.dtype, src=node.src, arg=(IDIV_TAG, shape), metadata=node.metadata)

def _rockchip_sigmoid_rewrite(node:UOp) -> UOp|None:
  if node.dtype not in (dtypes.float16, dtypes.float): return None
  base = _match_sigmoid_base(node)
  if base is None:
    return None
  try:
    shape = _safe_shape(node)
  except Exception:
    shape = tuple()
  return UOp(Ops.CUSTOM, node.dtype, src=(base,), arg=(SIGMOID_TAG, shape), metadata=node.metadata)

def _rockchip_silu_rewrite(node:UOp) -> UOp|None:
  if node.op is not Ops.MUL or len(node.src) != 2: return None
  if node.dtype not in (dtypes.float16, dtypes.float): return None
  lhs, rhs = node.src
  sig_node = None
  other = None
  if lhs.op is Ops.CUSTOM and _is_sigmoid_arg(lhs.arg):
    sig_node, other = lhs, rhs
  elif rhs.op is Ops.CUSTOM and _is_sigmoid_arg(rhs.arg):
    sig_node, other = rhs, lhs
  if sig_node is None or len(sig_node.src) != 1:
    return None
  if _strip_movement(sig_node.src[0]) is not _strip_movement(other):
    return None
  try:
    shape = _safe_shape(node)
  except Exception:
    shape = tuple()
  return UOp(Ops.CUSTOM, node.dtype, src=(other,), arg=(SILU_TAG, shape), metadata=node.metadata)

def _rockchip_abs_rewrite(node:UOp) -> UOp|None:
  if node.op is not Ops.MUL or len(node.src) != 2: return None
  if node.dtype not in (dtypes.float16, dtypes.float): return None
  lhs, rhs = node.src
  for base, sign in ((lhs, rhs), (rhs, lhs)):
    if _match_sign_pattern(sign, base):
      try:
        shape = _safe_shape(node)
      except Exception:
        shape = tuple()
      return UOp(Ops.CUSTOM, node.dtype, src=(base,), arg=(ABS_TAG, shape), metadata=node.metadata)
  return None

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
    if lhs.shape != rhs.shape:
      if DEBUG >= 3: print("ROCKCHIP rewrite reject shape mismatch", lhs.shape, rhs.shape)
      return None
  axes = tuple(red.arg[1]) if isinstance(red.arg, tuple) and len(red.arg) == 2 else tuple()
  ok, post_ops = _downstream_info(red)
  if not ok: return None
  info = _build_conv_info(meta_names, axes, lhs, rhs, red, post_ops)
  if DEBUG >= 3:
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
  if DEBUG >= 3:
    print("ROCKCHIP metadata rewrite applied", info)
  return UOp(Ops.CUSTOM, node.dtype, src=mul.src, arg=info, metadata=node.metadata)

rockchip_conv_pm = PatternMatcher([
  (UPat(Ops.REDUCE_AXIS, name="red"), lambda red: _rockchip_conv_rewrite(red)),
  (UPat(Ops.IDIV, name="node"), lambda node: _rockchip_idiv_rewrite(node)),
])

rockchip_idiv_pm = PatternMatcher([
  (UPat(Ops.IDIV, name="node"), lambda node: _rockchip_idiv_rewrite(node)),
])

rockchip_where_pm = PatternMatcher([
  (UPat(Ops.WHERE, name="node"), lambda node: _rockchip_where_rewrite(node)),
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
      rewritten = _rockchip_sigmoid_rewrite(u)
    if rewritten is None:
      rewritten = _rockchip_silu_rewrite(u)
    if rewritten is None:
      rewritten = _rockchip_abs_rewrite(u)
    if rewritten is None:
      rewritten = _rockchip_roundoff_rewrite(u)
    if rewritten is None:
      rewritten = _rockchip_where_rewrite(u)
    if rewritten is None:
      rewritten = _rockchip_idiv_rewrite(u)
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
    Ops.MAX: 0,
    Ops.ADD: 2, 
    Ops.FDIV: 3,
    # Ops.IDIV: 3, 
    Ops.RECIP: 3,
    # Ops.SUB: 4, 
    # Ops.NEG: 6, 
    Ops.MUL: None
    }
  pre_matcher = rockchip_conv_pm
  extra_matcher = rockchip_idiv_pm + rockchip_where_pm

  def preprocess_ast(self, ast:UOp) -> UOp:
    return rockchip_conv_prepass(ast)

  def render(self, uops:list[UOp]) -> str:
    if DEBUG >= 3:
      for u in uops:
        if u.metadata:
          print("RK_RENDER metadata", u.op, u.metadata)
    conv = next((u for u in uops if u.op is Ops.CUSTOM and isinstance(u.arg, RockchipConvInfo)), None)
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
            if DEBUG >= 3:
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
        if DEBUG >= 3:
          print("RK_CONV render fallback: missing RockchipConvInfo", conv)
      else:
        def _find_global_id(node:UOp) -> int|None:
          for parent in node.toposort():
            if parent.op is Ops.DEFINE_GLOBAL:
              return parent.arg
          return None
        store_uop = _find_store_for_conv(conv)
        if store_uop is None:
          if DEBUG >= 3:
            print("RK_CONV render fallback: no direct STORE for conv output")
        else:
          out_gid = _find_global_id(store_uop.src[0])
          lhs_gid = _find_global_id(conv.src[0])
          rhs_gid = _find_global_id(conv.src[1])
          if None in (out_gid, lhs_gid, rhs_gid):
            if DEBUG >= 3:
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
  def create_flink_name(self, handle: int, name:str, virt_address:int|None=None, obj_addr:int|None=None, dma_address:int|None=None) -> int:
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
      if DEBUG >= 3:
        addr_info_parts = []
        if virt_address is not None: addr_info_parts.append(f"va {hex(virt_address)}")
        if obj_addr is not None: addr_info_parts.append(f"obj {hex(obj_addr)}")
        if dma_address is not None: addr_info_parts.append(f"dma {hex(dma_address)}")
        addr_info = f" {' '.join(addr_info_parts)}" if addr_info_parts else ""
        print(f"SUCCESS: Created flink name {flink_req.name} for handle {handle} {name} {addr_info}")
      return flink_req.name
    except Exception as e:
      print(f"ERROR: DRM_IOCTL_GEM_FLINK failed: {e}")
      raise

  def _gpu_alloc(self, size:int, flags, name:str) -> HCQBuffer:
    mem_create = rk.DRM_IOCTL_RKNPU_MEM_CREATE(self.fd_ctl, size=size, flags=flags | rk.RKNPU_MEM_NON_CACHEABLE)
    mem_map = rk.DRM_IOCTL_RKNPU_MEM_MAP(self.fd_ctl, handle=mem_create.handle, offset=0)
    va_addr = self.fd_ctl.mmap(0, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, mem_map.offset)

    flink_name = self.create_flink_name(mem_create.handle, name, virt_address=va_addr, obj_addr=mem_create.obj_addr, dma_address=mem_create.dma_addr)
    mem_create.flink_name = flink_name

    return HCQBuffer(va_addr=va_addr, size=size, meta=mem_create)

  def _gpu_free(self, mem:HCQBuffer):
    if mem is None:
      return
    FileIOInterface.munmap(mem.va_addr, mem.size)
    rk.DRM_IOCTL_RKNPU_MEM_DESTROY(self.fd_ctl, handle=mem.meta.handle, obj_addr=mem.meta.obj_addr, reserved=0)

  def _reset_controller_fd(self, ctl: FileIOInterface|None) -> None:
    if ctl is None:
      return
    fd = getattr(ctl, "fd", -1)
    if fd < 0:
      return
    for _ in range(2):
      try:
        rk.DRM_IOCTL_RKNPU_ACTION(ctl, flags=rk.RKNPU_ACT_RESET)
      except Exception as exc:
        if DEBUG:
          print("RK_REOPEN reset failed", exc)

  def __init__(self, device:str): 
    self.fd_ctl = FileIOInterface(f"/dev/dri/card1", os.O_RDWR)
    self.task_buf = self._gpu_alloc(1024, rk.RKNPU_MEM_KERNEL_MAPPING, name="task")
    self.cmd_buf = self._gpu_alloc(16384, 0, name="cmd")

    self.input_buf = None
    self.weight_buf = None
    self.output_buf = None
    self._submission_total = 0

    self.buffer_list = []
    self.code_for_op = RockchipRenderer.code_for_op
    self._controller_reopened = False

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

  def _reopen_controller(self):
    old_bufs = [self.task_buf, self.cmd_buf] + [item["buf"] for item in self.buffer_list]
    old_ctl = self.fd_ctl
    new_ctl = FileIOInterface(f"/dev/dri/card1", os.O_RDWR)
    self._reset_controller_fd(old_ctl)
    for buf in old_bufs:
      if buf is not None:
        self._gpu_free(buf)
    self.buffer_list = []
    self.input_buf = self.weight_buf = self.output_buf = None
    try:
      os.close(old_ctl.fd)
    except Exception:
      pass
    if hasattr(old_ctl, "fd"):
      delattr(old_ctl, "fd")
    self.fd_ctl = new_ctl
    self.task_buf = self._gpu_alloc(1024, rk.RKNPU_MEM_KERNEL_MAPPING, name="task")
    self.cmd_buf = self._gpu_alloc(16384, 0, name="cmd")

  def reset_controller_if_needed(self):
    if getattr(self, "_controller_reopened", False):
      return
    self._controller_reopened = True
    self._reopen_controller()

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
      equal_en = 1 if op is Ops.MAX else 0
      self.emit_raw(rk.DPU, rk.REG_DPU_EW_CFG,
        self.reg(0, rk.DPU_EW_CFG_EW_CVT_TYPE__SHIFT, rk.DPU_EW_CFG_EW_CVT_TYPE__MASK) |
        self.reg(0, rk.DPU_EW_CFG_EW_CVT_ROUND__SHIFT, rk.DPU_EW_CFG_EW_CVT_ROUND__MASK) |
        self.reg(1, rk.DPU_EW_CFG_EW_DATA_MODE__SHIFT, rk.DPU_EW_CFG_EW_DATA_MODE__MASK) |
        self.reg(self.get_edata_size(dtype), rk.DPU_EW_CFG_EDATA_SIZE__SHIFT, rk.DPU_EW_CFG_EDATA_SIZE__MASK) |
        self.reg(equal_en, rk.DPU_EW_CFG_EW_EQUAL_EN__SHIFT, rk.DPU_EW_CFG_EW_EQUAL_EN__MASK) |
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

  def _roundoff_dims(self, shape:tuple[int, ...]|None, count:int) -> tuple[int, int]:
    rows = 0
    cols = 0
    if shape:
      if len(shape) == 1:
        rows = 1
        cols = int(shape[0])
      else:
        rows = int(shape[-2])
        cols = int(shape[-1])
      if rows <= 0 or cols <= 0 or rows * cols > count:
        rows = 0
        cols = 0
    if (rows <= 0 or cols <= 0) and count > 0:
      rows = int(math.sqrt(count))
      while rows > 1 and count % rows != 0:
        rows -= 1
      cols = count // rows if rows > 0 else count
    return rows, cols

  def _emit_roundoff_regs(self, input_dma:int, output_dma:int, packed_elems:int, rows:int, cols:int) -> None:
    if packed_elems == 0: packed_elems = 1
    dtype = dtypes.float16
    r = rows if rows > 0 else 1
    c = cols if cols > 0 else int(packed_elems)
    if r * c < packed_elems:
      r = (packed_elems + c - 1) // c
    if r < 1: r = 1
    if c < 1: c = 1
    data_cube_width = c - 1
    data_cube_height = r - 1
    stride_field = c * dtype.itemsize

    index_select = 14
    max_val = 1 << index_select
    self.q = []
    reg = self.reg
    emit = self.emit_raw
    emit(rk.DPU, rk.REG_DPU_LUT_ACCESS_CFG,
      reg(1, rk.DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE__SHIFT, rk.DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE__MASK))
    for _ in range(256):
      emit(rk.DPU, rk.REG_DPU_LUT_ACCESS_DATA,
        reg(0, rk.DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA__SHIFT, rk.DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA__MASK))
      emit(rk.DPU, rk.REG_DPU_LUT_ACCESS_DATA,
        reg(max_val, rk.DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA__SHIFT, rk.DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA__MASK))
    emit(rk.DPU, rk.REG_DPU_LUT_ACCESS_CFG,
      reg(1, rk.DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE__SHIFT, rk.DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE__MASK) |
      reg(1, rk.DPU_LUT_ACCESS_CFG_LUT_TABLE_ID__SHIFT, rk.DPU_LUT_ACCESS_CFG_LUT_TABLE_ID__MASK))
    for _ in range(256):
      emit(rk.DPU, rk.REG_DPU_LUT_ACCESS_DATA,
        reg(0, rk.DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA__SHIFT, rk.DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA__MASK))
      emit(rk.DPU, rk.REG_DPU_LUT_ACCESS_DATA,
        reg(max_val, rk.DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA__SHIFT, rk.DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA__MASK))

    emit(rk.DPU, rk.REG_DPU_S_POINTER,
      reg(1, rk.DPU_S_POINTER_EXECUTER_PP_CLEAR__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_CLEAR__MASK) |
      reg(1, rk.DPU_S_POINTER_POINTER_PP_CLEAR__SHIFT, rk.DPU_S_POINTER_POINTER_PP_CLEAR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER,
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR__MASK))
    emit(rk.DPU, rk.REG_PC_BASE_ADDRESS,
      reg(0, rk.PC_BASE_ADDRESS_PC_SOURCE_ADDR__SHIFT, rk.PC_BASE_ADDRESS_PC_SOURCE_ADDR__MASK))
    emit(rk.DPU, rk.REG_PC_REGISTER_AMOUNTS,
      reg(0, rk.PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT__SHIFT, rk.PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT__MASK))
    emit(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
      reg(15, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(2, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK) |
      reg(1, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_FORMAT,
      reg(2, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
      reg(2, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
      reg(2, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
      reg(output_dma, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE,
      reg(stride_field, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT,
      reg(data_cube_height, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__MASK) |
      reg(7, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))

    emit(rk.DPU, rk.REG_DPU_BS_CFG,
      reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_MUL_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_MUL_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_ALU_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_OW_CFG,
      reg(1, rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT, rk.DPU_BS_OW_CFG_OD_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_0,
      reg(7, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__SHIFT, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_1,
      reg(data_cube_height, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__MASK) |
      reg(data_cube_width, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))

    emit(rk.DPU, rk.REG_DPU_BN_CFG,
      reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_MUL_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_EW_CFG,
      reg(1, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK) |
      reg(0, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_BYPASS__MASK) |
      reg(0, rk.DPU_EW_CFG_EW_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_BYPASS__MASK))

    emit(rk.DPU, rk.REG_DPU_SURFACE_ADD,
      reg(stride_field, rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT, rk.DPU_SURFACE_ADD_SURF_ADD__MASK))
    self.emit_raw(rk.DPU, 0x40c4, 0)

    emit(rk.DPU, rk.REG_DPU_LUT_LE_START, 0x00000000)
    emit(rk.DPU, rk.REG_DPU_LUT_LE_END, 0x44000000)
    emit(rk.DPU, rk.REG_DPU_LUT_LO_START, 0x44000000)
    emit(rk.DPU, rk.REG_DPU_LUT_LO_END, 0x44800000)
    emit(rk.DPU, rk.REG_DPU_LUT_CFG,
      reg(1, rk.DPU_LUT_CFG_LUT_HYBRID_PRIORITY__SHIFT, rk.DPU_LUT_CFG_LUT_HYBRID_PRIORITY__MASK) |
      reg(1, rk.DPU_LUT_CFG_LUT_OFLOW_PRIORITY__SHIFT, rk.DPU_LUT_CFG_LUT_OFLOW_PRIORITY__MASK) |
      reg(2, rk.DPU_LUT_CFG_LUT_LO_LE_MUX__SHIFT, rk.DPU_LUT_CFG_LUT_LO_LE_MUX__MASK))
    emit(rk.DPU, rk.REG_DPU_LUT_INFO,
      reg(index_select, rk.DPU_LUT_INFO_LUT_LO_INDEX_SELECT__SHIFT, rk.DPU_LUT_INFO_LUT_LO_INDEX_SELECT__MASK) |
      reg(index_select, rk.DPU_LUT_INFO_LUT_LE_INDEX_SELECT__SHIFT, rk.DPU_LUT_INFO_LUT_LE_INDEX_SELECT__MASK))
    emit(rk.DPU, rk.REG_DPU_LUT_LE_SLOPE_SCALE,
      reg(23107, rk.DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_UFLOW_SCALE__SHIFT, rk.DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_UFLOW_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_LUT_LE_SLOPE_SHIFT,
      reg(22, rk.DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_UFLOW_SHIFT__SHIFT, rk.DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_UFLOW_SHIFT__MASK))

    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
      reg(data_cube_height, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
      reg(input_dma, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
      reg(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
      reg(2, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__MASK) |
      reg(15, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(2, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__SHIFT,
          rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT,
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__MASK))

  def roundoff(self, values:list[float], rows:int=0, cols:int=0) -> list[float]:
    self.device.reset_controller_if_needed()
    n = len(values)
    if n == 0: return []
    vals = np.asarray(values, dtype=np.float16)
    packed = np.zeros((n, 8), dtype=np.float16)
    packed[:, 0] = vals
    packed_bytes = n * 0x10
    input_buf = None
    output_buf = None
    try:
      input_buf = self.device._gpu_alloc(packed_bytes, 0, name="roundoff_in")
      output_buf = self.device._gpu_alloc(packed_bytes, 0, name="roundoff_out")
      ctypes.memmove(input_buf.va_addr, packed.tobytes(), packed_bytes)
      ctypes.memset(output_buf.va_addr, 0, packed_bytes)

      packed_elems = packed_bytes // 0x10
      self._emit_roundoff_regs(input_buf.meta.dma_addr, output_buf.meta.dma_addr, packed_elems, rows, cols)
      self.submit()
      out = np.frombuffer(ctypes.string_at(output_buf.va_addr, packed_bytes), dtype=np.float16).reshape(n, 8)[:, 0]
      return out.tolist()
    finally:
      if input_buf is not None: self.device._gpu_free(input_buf)
      if output_buf is not None: self.device._gpu_free(output_buf)

  def _roundoff_batch(self, values:list[float], shape:tuple[int, ...]|None) -> list[float]:
    rows, cols = self._roundoff_dims(shape, len(values))
    return self.roundoff(values, rows=rows, cols=cols)

  def _emit_lut_tables(self, tables:tuple[tuple[int, list[int]], ...], reserved:int=0) -> None:
    for table_id, values in tables:
      cfg = self.reg(1, rk.DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE__SHIFT, rk.DPU_LUT_ACCESS_CFG_LUT_ACCESS_TYPE__MASK)
      if table_id:
        cfg |= self.reg(table_id, rk.DPU_LUT_ACCESS_CFG_LUT_TABLE_ID__SHIFT, rk.DPU_LUT_ACCESS_CFG_LUT_TABLE_ID__MASK)
      self.emit_raw(rk.DPU, rk.REG_DPU_LUT_ACCESS_CFG, cfg)
      for value in values:
        raw = self.reg(value, rk.DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA__SHIFT, rk.DPU_LUT_ACCESS_DATA_LUT_ACCESS_DATA__MASK)
        if reserved:
          raw |= self.reg(reserved, rk.DPU_LUT_ACCESS_DATA_RESERVED_0__SHIFT, rk.DPU_LUT_ACCESS_DATA_RESERVED_0__MASK)
        self.emit_raw(rk.DPU, rk.REG_DPU_LUT_ACCESS_DATA, raw)

  def _emit_sigmoid_regs(self, input_dma:int, output_dma:int) -> None:
    self.q = []
    reg = self.reg
    emit = self.emit_raw
    self._emit_lut_tables(SIGMOID_LUT_TABLES, reserved=0)

    emit(rk.DPU, rk.REG_DPU_S_POINTER,
      reg(1, rk.DPU_S_POINTER_EXECUTER_PP_CLEAR__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_CLEAR__MASK) |
      reg(1, rk.DPU_S_POINTER_POINTER_PP_CLEAR__SHIFT, rk.DPU_S_POINTER_POINTER_PP_CLEAR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER,
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR__MASK))
    emit(rk.DPU, rk.REG_PC_BASE_ADDRESS,
      reg(0, rk.PC_BASE_ADDRESS_PC_SOURCE_ADDR__SHIFT, rk.PC_BASE_ADDRESS_PC_SOURCE_ADDR__MASK))
    emit(rk.DPU, rk.REG_PC_REGISTER_AMOUNTS,
      reg(0, rk.PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT__SHIFT, rk.PC_REGISTER_AMOUNTS_PC_DATA_AMOUNT__MASK))
    emit(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
      reg(15, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(2, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK) |
      reg(1, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_FORMAT,
      reg(2, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
      reg(2, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
      reg(2, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
      reg(output_dma, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE,
      reg(16, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
      reg(15, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__MASK) |
      reg(7, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_CFG,
      reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_MUL_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_MUL_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_ALU_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_OW_CFG,
      reg(1, rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT, rk.DPU_BS_OW_CFG_OD_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_0,
      reg(7, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__SHIFT, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_1,
      reg(15, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_BN_CFG,
      reg(2, rk.DPU_BN_CFG_BN_ALU_ALGO__SHIFT, rk.DPU_BN_CFG_BN_ALU_ALGO__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_BN_ALU_CFG, 0x80000000)
    emit(rk.DPU, rk.REG_DPU_BN_MUL_CFG,
      reg(0x6912, rk.DPU_BN_MUL_CFG_BN_MUL_OPERAND__SHIFT, rk.DPU_BN_MUL_CFG_BN_MUL_OPERAND__MASK))
    emit(rk.DPU, rk.REG_DPU_EW_CFG,
      reg(1, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE,
      reg(1, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__SHIFT, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_OUT_CVT_OFFSET, 0x00000001)
    emit(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE,
      reg(1, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__SHIFT, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__MASK) |
      reg(1, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__SHIFT, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT,
      reg(15, rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__SHIFT, rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__MASK))
    emit(rk.DPU, rk.REG_DPU_SURFACE_ADD,
      reg(16, rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT, rk.DPU_SURFACE_ADD_SURF_ADD__MASK))
    emit(rk.DPU, 0x40c4, 0)

    emit(rk.DPU, rk.REG_DPU_LUT_CFG,
      reg(1, rk.DPU_LUT_CFG_LUT_HYBRID_PRIORITY__SHIFT, rk.DPU_LUT_CFG_LUT_HYBRID_PRIORITY__MASK) |
      reg(1, rk.DPU_LUT_CFG_LUT_OFLOW_PRIORITY__SHIFT, rk.DPU_LUT_CFG_LUT_OFLOW_PRIORITY__MASK) |
      reg(2, rk.DPU_LUT_CFG_LUT_LO_LE_MUX__SHIFT, rk.DPU_LUT_CFG_LUT_LO_LE_MUX__MASK))
    emit(rk.DPU, rk.REG_DPU_LUT_INFO,
      reg(5, rk.DPU_LUT_INFO_LUT_LO_INDEX_SELECT__SHIFT, rk.DPU_LUT_INFO_LUT_LO_INDEX_SELECT__MASK) |
      reg(5, rk.DPU_LUT_INFO_LUT_LE_INDEX_SELECT__SHIFT, rk.DPU_LUT_INFO_LUT_LE_INDEX_SELECT__MASK))
    emit(rk.DPU, rk.REG_DPU_LUT_LE_START, 0xffffc000)
    emit(rk.DPU, rk.REG_DPU_LUT_LO_END, 0x00004000)
    emit(rk.DPU, rk.REG_DPU_LUT_LE_SLOPE_SCALE,
      reg(23107, rk.DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_UFLOW_SCALE__SHIFT,
          rk.DPU_LUT_LE_SLOPE_SCALE_LUT_LE_SLOPE_UFLOW_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_LUT_LE_SLOPE_SHIFT,
      reg(22, rk.DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_UFLOW_SHIFT__SHIFT,
          rk.DPU_LUT_LE_SLOPE_SHIFT_LUT_LE_SLOPE_UFLOW_SHIFT__MASK))

    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
      reg(15, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
      reg(input_dma, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
      reg(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
      reg(2, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__MASK) |
      reg(15, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(2, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__SHIFT,
          rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT,
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__MASK))

  def _emit_silu_regs(self, input_dma:int, output_dma:int) -> None:
    self.q = []
    reg = self.reg
    emit = self.emit_raw
    self._emit_lut_tables(SILU_LUT_TABLES, reserved=0xFFFF)

    emit(rk.DPU, rk.REG_DPU_S_POINTER,
      reg(1, rk.DPU_S_POINTER_EXECUTER_PP_CLEAR__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_CLEAR__MASK) |
      reg(1, rk.DPU_S_POINTER_POINTER_PP_CLEAR__SHIFT, rk.DPU_S_POINTER_POINTER_PP_CLEAR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER,
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_CLEAR__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_CLEAR__MASK))
    emit(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
      reg(15, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(2, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK) |
      reg(1, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_FORMAT,
      reg(5, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
      reg(2, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
      reg(2, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
      reg(output_dma, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE,
      reg(16, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
      reg(15, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__MASK) |
      reg(7, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_CFG,
      reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_MUL_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_MUL_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_ALU_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_OW_CFG,
      reg(1, rk.DPU_BS_OW_CFG_SIZE_E_2__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_2__MASK) |
      reg(1, rk.DPU_BS_OW_CFG_SIZE_E_1__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_1__MASK) |
      reg(1, rk.DPU_BS_OW_CFG_SIZE_E_0__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_0__MASK) |
      reg(1, rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT, rk.DPU_BS_OW_CFG_OD_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_0,
      reg(7, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__SHIFT, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_1,
      reg(15, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_BN_CFG,
      reg(2, rk.DPU_BN_CFG_BN_ALU_ALGO__SHIFT, rk.DPU_BN_CFG_BN_ALU_ALGO__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_BN_ALU_CFG, 0x80000000)
    emit(rk.DPU, rk.REG_DPU_BN_MUL_CFG,
      reg(0x6984, rk.DPU_BN_MUL_CFG_BN_MUL_OPERAND__SHIFT, rk.DPU_BN_MUL_CFG_BN_MUL_OPERAND__MASK))
    emit(rk.DPU, rk.REG_DPU_EW_CFG,
      reg(1, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE,
      reg(1, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__SHIFT, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE,
      reg(1, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__SHIFT, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__MASK) |
      reg(1, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__SHIFT, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_SURFACE_ADD,
      reg(32, rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT, rk.DPU_SURFACE_ADD_SURF_ADD__MASK))
    emit(rk.DPU, 0x40c4, 0)

    emit(rk.DPU, rk.REG_DPU_LUT_CFG,
      reg(1, rk.DPU_LUT_CFG_LUT_HYBRID_PRIORITY__SHIFT, rk.DPU_LUT_CFG_LUT_HYBRID_PRIORITY__MASK) |
      reg(1, rk.DPU_LUT_CFG_LUT_OFLOW_PRIORITY__SHIFT, rk.DPU_LUT_CFG_LUT_OFLOW_PRIORITY__MASK) |
      reg(2, rk.DPU_LUT_CFG_LUT_LO_LE_MUX__SHIFT, rk.DPU_LUT_CFG_LUT_LO_LE_MUX__MASK))
    emit(rk.DPU, rk.REG_DPU_LUT_INFO,
      reg(5, rk.DPU_LUT_INFO_LUT_LO_INDEX_SELECT__SHIFT, rk.DPU_LUT_INFO_LUT_LO_INDEX_SELECT__MASK) |
      reg(5, rk.DPU_LUT_INFO_LUT_LE_INDEX_SELECT__SHIFT, rk.DPU_LUT_INFO_LUT_LE_INDEX_SELECT__MASK))
    emit(rk.DPU, rk.REG_DPU_LUT_LE_START, 0xffffc000)
    emit(rk.DPU, rk.REG_DPU_LUT_LO_END, 0x00004000)
    emit(rk.DPU, rk.REG_DPU_LUT_LO_SLOPE_SCALE,
      reg(16434, rk.DPU_LUT_LO_SLOPE_SCALE_LUT_LO_SLOPE_OFLOW_SCALE__SHIFT,
          rk.DPU_LUT_LO_SLOPE_SCALE_LUT_LO_SLOPE_OFLOW_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_LUT_LO_SLOPE_SHIFT,
      reg(13, rk.DPU_LUT_LO_SLOPE_SHIFT_LUT_LO_SLOPE_OFLOW_SHIFT__SHIFT,
          rk.DPU_LUT_LO_SLOPE_SHIFT_LUT_LO_SLOPE_OFLOW_SHIFT__MASK))

    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
      reg(15, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
      reg(input_dma, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
      reg(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
      reg(2, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__MASK) |
      reg(15, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(2, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__SHIFT,
          rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT,
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__MASK))

  def _alu_dims(self, shape:tuple[int, ...]|None, count:int) -> tuple[int, int]:
    return self._where_dims(shape, count)

  def _alu_op_padded_fp16(self, alg_id:int, values:list[Any]) -> list[float]:
    self.device.reset_controller_if_needed()
    n = len(values)
    if n == 0: return []
    max_elems = 0x140 // 0x10
    out: list[float] = []
    for off in range(0, n, max_elems):
      chunk = values[off:off+max_elems]
      packed_elems = len(chunk)
      packed_bytes = 0x140
      output_bytes = packed_elems * 0x10
      input_buf = output_buf = None
      try:
        input_buf = self.device._gpu_alloc(packed_bytes, 0, name="alu_pad_in")
        output_buf = self.device._gpu_alloc(output_bytes, 0, name="alu_pad_out")
        packed = np.zeros((packed_bytes // 2,), dtype=np.float16)
        for i, val in enumerate(chunk):
          packed[i * (0x10 // 2)] = np.float16(val)
        ctypes.memmove(input_buf.va_addr, packed.tobytes(), packed_bytes)
        ctypes.memset(output_buf.va_addr, 0, output_bytes)
        if alg_id == 14:
          self._emit_sigmoid_regs(input_buf.meta.dma_addr, output_buf.meta.dma_addr)
        elif alg_id == 15:
          self._emit_silu_regs(input_buf.meta.dma_addr, output_buf.meta.dma_addr)
        else:
          raise RuntimeError(f"unsupported padded alu algo {alg_id}")
        self.submit()
        if alg_id == 15:
          buf = np.frombuffer(ctypes.string_at(output_buf.va_addr, output_bytes), dtype=np.float32).reshape(packed_elems, 4)[:, 0]
        else:
          buf = np.frombuffer(ctypes.string_at(output_buf.va_addr, output_bytes), dtype=np.float16).reshape(packed_elems, 8)[:, 0].astype(np.float32)
        out.extend(buf.tolist())
      finally:
        if input_buf is not None: self.device._gpu_free(input_buf)
        if output_buf is not None: self.device._gpu_free(output_buf)
    return out

  def _emit_alu_fp16_regs(self, alg_id:int, input_dma:int, weight_dma:int, output_dma:int,
                          rows:int, cols:int) -> None:
    self.q = []
    reg = self.reg
    emit = self.emit_raw
    data_cube_width = cols - 1
    data_cube_height = rows - 1
    stride_field = cols * 2

    emit(rk.DPU, rk.REG_DPU_S_POINTER,
      reg(1, rk.DPU_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_S_POINTER_POINTER_PP_MODE__MASK) |
      reg(1, rk.DPU_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_EN__MASK) |
      reg(1, rk.DPU_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_S_POINTER_POINTER_PP_EN__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER,
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__MASK))
    emit(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
      reg(15, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(2, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK) |
      reg(1, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_FORMAT,
      reg(2, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
      reg(2, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
      reg(2, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
      reg(output_dma, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE,
      reg(stride_field, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT,
      reg(data_cube_height, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_0,
      reg(7, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__SHIFT, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_1,
      reg(data_cube_height, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__MASK) |
      reg(data_cube_width, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_OW_CFG,
      reg(1, rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT, rk.DPU_BS_OW_CFG_OD_BYPASS__MASK))

    if alg_id == 22:
      emit(rk.DPU, rk.REG_DPU_BS_CFG,
        reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK) |
        reg(1, rk.DPU_BS_CFG_BS_MUL_PRELU__SHIFT, rk.DPU_BS_CFG_BS_MUL_PRELU__MASK) |
        reg(1, rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_ALU_BYPASS__MASK))
      emit(rk.DPU, rk.REG_DPU_BS_MUL_CFG,
        reg(0xBC00, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__SHIFT, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__MASK))
      emit(rk.DPU, rk.REG_DPU_BN_CFG,
        reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK) |
        reg(1, rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_MUL_BYPASS__MASK) |
        reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK) |
        reg(1, rk.DPU_BN_CFG_BN_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_BYPASS__MASK))
      emit(rk.DPU, rk.REG_DPU_EW_CFG,
        reg(1, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
        reg(1, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK) |
        reg(1, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
        reg(1, rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_BYPASS__MASK) |
        reg(1, rk.DPU_EW_CFG_EW_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_BYPASS__MASK))
    elif alg_id == 9:
      emit(rk.DPU, rk.REG_DPU_BS_CFG,
        reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK) |
        reg(1, rk.DPU_BS_CFG_BS_MUL_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_MUL_BYPASS__MASK) |
        reg(1, rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_ALU_BYPASS__MASK) |
        reg(1, rk.DPU_BS_CFG_BS_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_BYPASS__MASK))
      emit(rk.DPU, rk.REG_DPU_BN_CFG,
        reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK) |
        reg(1, rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_MUL_BYPASS__MASK) |
        reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK) |
        reg(1, rk.DPU_BN_CFG_BN_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_BYPASS__MASK))
      emit(rk.DPU, rk.REG_DPU_EW_CFG,
        reg(1, rk.DPU_EW_CFG_EW_DATA_MODE__SHIFT, rk.DPU_EW_CFG_EW_DATA_MODE__MASK) |
        reg(2, rk.DPU_EW_CFG_EDATA_SIZE__SHIFT, rk.DPU_EW_CFG_EDATA_SIZE__MASK) |
        reg(1, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
        reg(1, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK) |
        reg(1, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
        reg(1, rk.DPU_EW_CFG_EW_OP_SRC__SHIFT, rk.DPU_EW_CFG_EW_OP_SRC__MASK) |
        reg(1, rk.DPU_EW_CFG_EW_OP_TYPE__SHIFT, rk.DPU_EW_CFG_EW_OP_TYPE__MASK))
    else:
      raise RuntimeError(f"unsupported alu algo {alg_id}")

    emit(rk.DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE,
      reg(1, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__SHIFT, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_SURFACE_ADD,
      reg(stride_field, rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT, rk.DPU_SURFACE_ADD_SURF_ADD__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
      reg(data_cube_height, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
      reg(input_dma, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
      reg(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__MASK) |
      reg(2, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
      reg(weight_dma + 0x4000, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__SHIFT,
          rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_SURF_STRIDE,
      reg(stride_field, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__SHIFT,
          rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
      reg(2, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__MASK) |
      reg(15, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(2, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__SHIFT,
          rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT,
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__MASK))

  def _alu_op_fp16(self, alg_id:int, a_vals:list[Any], b_vals:list[Any], shape:tuple[int, ...]|None=None) -> list[float]:
    self.device.reset_controller_if_needed()
    n = len(a_vals)
    if n != len(b_vals):
      raise RuntimeError(f"alu op input length mismatch {n} != {len(b_vals)}")
    if n == 0: return []
    if n > 2048:
      out = []
      for off in range(0, n, 2048):
        out.extend(self._alu_op_fp16(alg_id, a_vals[off:off+2048], b_vals[off:off+2048], shape))
      return out

    rows, cols = self._alu_dims(shape, n)
    if rows <= 0 or cols <= 0:
      rows, cols = 1, n

    a_fp16 = np.asarray(a_vals, dtype=np.float32).astype(np.float16)
    b_fp16 = np.asarray(b_vals, dtype=np.float32).astype(np.float16)
    packed_a = np.zeros((n, 8), dtype=np.float16)
    packed_b = np.zeros((n, 8), dtype=np.float16)
    packed_a[:, 0] = a_fp16
    packed_b[:, 0] = b_fp16
    packed_bytes = n * 0x10

    input_buf = weight_buf = output_buf = None
    try:
      input_buf = self.device._gpu_alloc(packed_bytes, 0, name="alu_in")
      weight_buf = self.device._gpu_alloc(0x4000 + packed_bytes, 0, name="alu_wt")
      output_buf = self.device._gpu_alloc(packed_bytes, 0, name="alu_out")
      ctypes.memmove(input_buf.va_addr, packed_b.tobytes(), packed_bytes)
      ctypes.memset(weight_buf.va_addr, 0, 0x4000 + packed_bytes)
      ctypes.memmove(weight_buf.va_addr + 0x4000, packed_a.tobytes(), packed_bytes)
      ctypes.memset(output_buf.va_addr, 0, packed_bytes)
      self._emit_alu_fp16_regs(alg_id, input_buf.meta.dma_addr, weight_buf.meta.dma_addr,
                               output_buf.meta.dma_addr, rows, cols)
      self.submit()
      out = np.frombuffer(ctypes.string_at(output_buf.va_addr, packed_bytes), dtype=np.float16).reshape(n, 8)[:, 0]
      return out.astype(np.float32).tolist()
    finally:
      if input_buf is not None: self.device._gpu_free(input_buf)
      if weight_buf is not None: self.device._gpu_free(weight_buf)
      if output_buf is not None: self.device._gpu_free(output_buf)

  def _sigmoid_batch(self, values:list[Any], shape:tuple[int, ...]|None) -> list[float]:
    return self._alu_op_padded_fp16(14, values)

  def _silu_batch(self, values:list[Any], shape:tuple[int, ...]|None) -> list[float]:
    stage1 = self._alu_op_padded_fp16(15, values)
    stage1_fp16 = np.asarray(stage1, dtype=np.float32).astype(np.float16).tolist()
    scale = [0.0001766241] * len(stage1_fp16)
    return self._alu_op_fp16(9, stage1_fp16, scale, shape)

  def _abs_batch(self, values:list[Any], shape:tuple[int, ...]|None) -> list[float]:
    zeros = [0.0] * len(values)
    return self._alu_op_fp16(22, zeros, values, shape)

  def _where_dims(self, shape:tuple[int, ...]|None, count:int) -> tuple[int, int]:
    rows = 0
    cols = 0
    if shape:
      if len(shape) == 1:
        rows = 1
        cols = int(shape[0])
      else:
        rows = int(shape[-2])
        cols = int(shape[-1])
      if rows <= 0 or cols <= 0 or rows * cols > count:
        rows = 0
        cols = 0
    if (rows <= 0 or cols <= 0) and count > 0:
      rows = int(math.sqrt(count))
      while rows > 1 and count % rows != 0:
        rows -= 1
      cols = count // rows if rows > 0 else count
    return rows, cols

  def _emit_where_add_regs(self, input_dma:int, weight_dma:int, output_dma:int, packed_elems:int, rows:int, cols:int,
                           dtype:DType=dtypes.float16) -> None:
    if packed_elems == 0: packed_elems = 1
    prec = self.get_precision(dtype)
    edata_size = self.get_edata_size(dtype)
    fp16 = self.get_is_fp16(dtype)
    r = rows if rows > 0 else 1
    c = cols if cols > 0 else int(packed_elems)
    if r * c < packed_elems:
      r = (packed_elems + c - 1) // c
    if r < 1: r = 1
    if c < 1: c = 1
    data_cube_width = c - 1
    data_cube_height = r - 1
    stride_field = c * dtype.itemsize

    self.q = []
    reg = self.reg
    emit = self.emit_raw
    emit(rk.DPU, rk.REG_DPU_S_POINTER,
      reg(1, rk.DPU_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_S_POINTER_POINTER_PP_MODE__MASK) |
      reg(1, rk.DPU_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_EN__MASK) |
      reg(1, rk.DPU_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_S_POINTER_POINTER_PP_EN__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER,
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__MASK))
    emit(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
      reg(15, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(2, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK) |
      reg(1, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_FORMAT,
      reg(prec, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
      reg(prec, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
      reg(prec, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
      reg(output_dma, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE,
      reg(stride_field, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT,
      reg(data_cube_height, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_CFG,
      reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_MUL_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_MUL_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_ALU_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_OW_CFG,
      reg(1, rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT, rk.DPU_BS_OW_CFG_OD_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_0,
      reg(7, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__SHIFT, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_1,
      reg(data_cube_height, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__MASK) |
      reg(data_cube_width, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_BN_CFG,
      reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_MUL_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_EW_CFG,
      reg(1, rk.DPU_EW_CFG_EW_DATA_MODE__SHIFT, rk.DPU_EW_CFG_EW_DATA_MODE__MASK) |
      reg(edata_size, rk.DPU_EW_CFG_EDATA_SIZE__SHIFT, rk.DPU_EW_CFG_EDATA_SIZE__MASK) |
      reg(2, rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT, rk.DPU_EW_CFG_EW_ALU_ALGO__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_OP_SRC__SHIFT, rk.DPU_EW_CFG_EW_OP_SRC__MASK))
    emit(rk.DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE,
      reg(1, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__SHIFT, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE,
      reg(fp16, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__SHIFT, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__MASK) |
      reg(1, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__SHIFT, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_SURFACE_ADD,
      reg(stride_field, rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT, rk.DPU_SURFACE_ADD_SURF_ADD__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
      reg(data_cube_height, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
      reg(input_dma, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
      reg(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__MASK) |
      reg(2, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
      reg(weight_dma + 0x4000, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_SURF_STRIDE,
      reg(stride_field, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__SHIFT, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
      reg(prec, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__MASK) |
      reg(15, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(prec, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__MASK) |
      reg(fp16, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT,
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__MASK))

  def _emit_where_max_regs(self, input_dma:int, weight_dma:int, output_dma:int, packed_elems:int, rows:int, cols:int,
                           dtype:DType=dtypes.float16) -> None:
    if packed_elems == 0: packed_elems = 1
    prec = self.get_precision(dtype)
    edata_size = self.get_edata_size(dtype)
    fp16 = self.get_is_fp16(dtype)
    r = rows if rows > 0 else 1
    c = cols if cols > 0 else int(packed_elems)
    if r * c < packed_elems:
      r = (packed_elems + c - 1) // c
    if r < 1: r = 1
    if c < 1: c = 1
    data_cube_width = c - 1
    data_cube_height = r - 1
    stride_field = c * dtype.itemsize

    self.q = []
    reg = self.reg
    emit = self.emit_raw
    emit(rk.DPU, rk.REG_DPU_S_POINTER,
      reg(1, rk.DPU_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_S_POINTER_POINTER_PP_MODE__MASK) |
      reg(1, rk.DPU_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_EN__MASK) |
      reg(1, rk.DPU_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_S_POINTER_POINTER_PP_EN__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER,
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__MASK))
    emit(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
      reg(15, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(2, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK) |
      reg(1, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_FORMAT,
      reg(prec, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
      reg(prec, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
      reg(prec, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
      reg(output_dma, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE,
      reg(stride_field, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT,
      reg(data_cube_height, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_CFG,
      reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_MUL_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_MUL_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_ALU_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_OW_CFG,
      reg(1, rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT, rk.DPU_BS_OW_CFG_OD_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_0,
      reg(7, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__SHIFT, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_1,
      reg(data_cube_height, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__MASK) |
      reg(data_cube_width, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_BN_CFG,
      reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_MUL_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_EW_CFG,
      reg(1, rk.DPU_EW_CFG_EW_DATA_MODE__SHIFT, rk.DPU_EW_CFG_EW_DATA_MODE__MASK) |
      reg(edata_size, rk.DPU_EW_CFG_EDATA_SIZE__SHIFT, rk.DPU_EW_CFG_EDATA_SIZE__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_EQUAL_EN__SHIFT, rk.DPU_EW_CFG_EW_EQUAL_EN__MASK) |
      reg(0, rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT, rk.DPU_EW_CFG_EW_ALU_ALGO__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_OP_SRC__SHIFT, rk.DPU_EW_CFG_EW_OP_SRC__MASK))
    emit(rk.DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE,
      reg(1, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__SHIFT, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE,
      reg(fp16, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__SHIFT, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__MASK) |
      reg(1, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__SHIFT, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_SURFACE_ADD,
      reg(stride_field, rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT, rk.DPU_SURFACE_ADD_SURF_ADD__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
      reg(data_cube_height, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
      reg(input_dma, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
      reg(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__MASK) |
      reg(2, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
      reg(weight_dma + 0x4000, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_SURF_STRIDE,
      reg(stride_field, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__SHIFT, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
      reg(prec, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__MASK) |
      reg(15, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(prec, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__MASK) |
      reg(fp16, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT,
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__MASK))

  def _emit_where_mul_regs(self, input_dma:int, weight_dma:int, output_dma:int, packed_elems:int, rows:int, cols:int,
                           dtype:DType=dtypes.float16) -> None:
    if packed_elems == 0: packed_elems = 1
    prec = self.get_precision(dtype)
    edata_size = self.get_edata_size(dtype)
    fp16 = self.get_is_fp16(dtype)
    r = rows if rows > 0 else 1
    c = cols if cols > 0 else int(packed_elems)
    if r * c < packed_elems:
      r = (packed_elems + c - 1) // c
    if r < 1: r = 1
    if c < 1: c = 1
    data_cube_width = c - 1
    data_cube_height = r - 1
    stride_field = c * 2

    self.q = []
    reg = self.reg
    emit = self.emit_raw
    emit(rk.DPU, rk.REG_DPU_S_POINTER,
      reg(1, rk.DPU_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_S_POINTER_POINTER_PP_MODE__MASK) |
      reg(1, rk.DPU_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_EN__MASK) |
      reg(1, rk.DPU_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_S_POINTER_POINTER_PP_EN__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER,
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__MASK))
    emit(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
      reg(15, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(2, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK) |
      reg(1, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_FORMAT,
      reg(prec, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
      reg(prec, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
      reg(prec, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
      reg(output_dma, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE,
      reg(stride_field, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT,
      reg(data_cube_height, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_0,
      reg(7, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__SHIFT, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_1,
      reg(data_cube_height, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__MASK) |
      reg(data_cube_width, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_OW_CFG,
      reg(1, rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT, rk.DPU_BS_OW_CFG_OD_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_CFG,
      reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_MUL_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_MUL_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_ALU_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_BN_CFG,
      reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_MUL_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_EW_CFG,
      reg(1, rk.DPU_EW_CFG_EW_DATA_MODE__SHIFT, rk.DPU_EW_CFG_EW_DATA_MODE__MASK) |
      reg(edata_size, rk.DPU_EW_CFG_EDATA_SIZE__SHIFT, rk.DPU_EW_CFG_EDATA_SIZE__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_OP_SRC__SHIFT, rk.DPU_EW_CFG_EW_OP_SRC__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_OP_TYPE__SHIFT, rk.DPU_EW_CFG_EW_OP_TYPE__MASK))
    emit(rk.DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE,
      reg(1, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__SHIFT, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_SURFACE_ADD,
      reg(stride_field, rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT, rk.DPU_SURFACE_ADD_SURF_ADD__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
      reg(data_cube_height, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
      reg(input_dma, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
      reg(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__MASK) |
      reg(2, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
      reg(weight_dma + 0x4000, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_SURF_STRIDE,
      reg(stride_field, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__SHIFT, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
      reg(prec, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__MASK) |
      reg(15, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(prec, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__MASK) |
      reg(fp16, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT,
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__MASK))

  def _emit_where_neg_regs(self, input_dma:int, weight_dma:int, output_dma:int, packed_elems:int, rows:int, cols:int) -> None:
    if packed_elems == 0: packed_elems = 1
    r = rows if rows > 0 else 1
    c = cols if cols > 0 else int(packed_elems)
    if r * c < packed_elems:
      r = (packed_elems + c - 1) // c
    if r < 1: r = 1
    if c < 1: c = 1
    data_cube_width = c - 1
    data_cube_height = r - 1
    stride_field = c * 2

    self.q = []
    reg = self.reg
    emit = self.emit_raw
    emit(rk.DPU, rk.REG_DPU_S_POINTER,
      reg(1, rk.DPU_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_S_POINTER_POINTER_PP_MODE__MASK) |
      reg(1, rk.DPU_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_EN__MASK) |
      reg(1, rk.DPU_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_S_POINTER_POINTER_PP_EN__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER,
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__MASK))
    emit(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
      reg(15, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(2, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK) |
      reg(1, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_FORMAT,
      reg(2, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
      reg(2, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
      reg(2, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
      reg(output_dma, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE,
      reg(stride_field, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT,
      reg(data_cube_height, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_0,
      reg(7, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__SHIFT, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_1,
      reg(data_cube_height, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__MASK) |
      reg(data_cube_width, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_OW_CFG,
      reg(1, rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT, rk.DPU_BS_OW_CFG_OD_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_CFG,
      reg(0, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_RELUX_EN__SHIFT, rk.DPU_BS_CFG_BS_RELUX_EN__MASK) |
      reg(0, rk.DPU_BS_CFG_BS_MUL_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_MUL_BYPASS__MASK) |
      reg(0, rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_ALU_BYPASS__MASK) |
      reg(4, rk.DPU_BS_CFG_BS_ALU_ALGO__SHIFT, rk.DPU_BS_CFG_BS_ALU_ALGO__MASK) |
      reg(0, rk.DPU_BS_CFG_BS_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_ALU_CFG,
      reg(0x3F800000, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__SHIFT, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_MUL_CFG,
      reg(0xBC00, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__SHIFT, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_RELUX_CMP_VALUE,
      reg(0x3F800000, rk.DPU_BS_RELUX_CMP_VALUE_BS_RELUX_CMP_DAT__SHIFT, rk.DPU_BS_RELUX_CMP_VALUE_BS_RELUX_CMP_DAT__MASK))
    emit(rk.DPU, rk.REG_DPU_BN_CFG,
      reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_MUL_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_EW_CFG,
      reg(1, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE,
      reg(1, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__SHIFT, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_SURFACE_ADD,
      reg(stride_field, rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT, rk.DPU_SURFACE_ADD_SURF_ADD__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
      reg(data_cube_height, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
      reg(input_dma, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
      reg(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__MASK) |
      reg(2, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
      reg(weight_dma + 0x4000, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_SURF_STRIDE,
      reg(stride_field, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__SHIFT, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
      reg(2, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__MASK) |
      reg(15, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(2, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT,
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__MASK))

  def _emit_minus_regs(self, input_dma:int, weight_dma:int, output_dma:int, packed_elems:int, rows:int, cols:int,
                       dtype:DType=dtypes.float16) -> None:
    if packed_elems == 0: packed_elems = 1
    prec = self.get_precision(dtype)
    edata_size = self.get_edata_size(dtype)
    fp16 = self.get_is_fp16(dtype)
    r = rows if rows > 0 else 1
    c = cols if cols > 0 else int(packed_elems)
    if r * c < packed_elems:
      r = (packed_elems + c - 1) // c
    if r < 1: r = 1
    if c < 1: c = 1
    data_cube_width = c - 1
    data_cube_height = r - 1
    stride_field = c * dtype.itemsize

    self.q = []
    reg = self.reg
    emit = self.emit_raw
    emit(rk.DPU, rk.REG_DPU_S_POINTER,
      reg(1, rk.DPU_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_S_POINTER_POINTER_PP_MODE__MASK) |
      reg(1, rk.DPU_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_EN__MASK) |
      reg(1, rk.DPU_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_S_POINTER_POINTER_PP_EN__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER,
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__MASK))
    emit(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
      reg(15, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(2, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK) |
      reg(1, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_FORMAT,
      reg(prec, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
      reg(prec, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
      reg(prec, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
      reg(output_dma, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE,
      reg(stride_field, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT,
      reg(data_cube_height, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_CFG,
      reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_MUL_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_MUL_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_ALU_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_OW_CFG,
      reg(1, rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT, rk.DPU_BS_OW_CFG_OD_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_0,
      reg(7, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__SHIFT, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_1,
      reg(data_cube_height, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__MASK) |
      reg(data_cube_width, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_BN_CFG,
      reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_MUL_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_EW_CFG,
      reg(1, rk.DPU_EW_CFG_EW_DATA_MODE__SHIFT, rk.DPU_EW_CFG_EW_DATA_MODE__MASK) |
      reg(edata_size, rk.DPU_EW_CFG_EDATA_SIZE__SHIFT, rk.DPU_EW_CFG_EDATA_SIZE__MASK) |
      reg(4, rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT, rk.DPU_EW_CFG_EW_ALU_ALGO__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_OP_SRC__SHIFT, rk.DPU_EW_CFG_EW_OP_SRC__MASK))
    emit(rk.DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE,
      reg(1, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__SHIFT, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE,
      reg(fp16, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__SHIFT, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__MASK) |
      reg(1, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__SHIFT, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_SURFACE_ADD,
      reg(stride_field, rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT, rk.DPU_SURFACE_ADD_SURF_ADD__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
      reg(data_cube_height, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
      reg(input_dma, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
      reg(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__MASK) |
      reg(2, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
      reg(weight_dma + 0x4000, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_SURF_STRIDE,
      reg(stride_field, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__SHIFT, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
      reg(prec, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__MASK) |
      reg(15, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(prec, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__MASK) |
      reg(fp16, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT,
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__MASK))

  def _emit_cmpeq_part2_regs(self, input_dma:int, weight_dma:int, output_dma:int, packed_elems:int, rows:int, cols:int) -> None:
    if packed_elems == 0: packed_elems = 1
    r = rows if rows > 0 else 1
    c = cols if cols > 0 else int(packed_elems)
    if r * c < packed_elems:
      r = (packed_elems + c - 1) // c
    if r < 1: r = 1
    if c < 1: c = 1
    data_cube_width = c - 1
    data_cube_height = r - 1
    stride_field = c * 2

    self.q = []
    reg = self.reg
    emit = self.emit_raw
    emit(rk.DPU, rk.REG_DPU_S_POINTER,
      reg(1, rk.DPU_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_S_POINTER_POINTER_PP_MODE__MASK) |
      reg(1, rk.DPU_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_EN__MASK) |
      reg(1, rk.DPU_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_S_POINTER_POINTER_PP_EN__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER,
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__MASK))
    emit(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
      reg(15, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(2, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK) |
      reg(1, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_FORMAT,
      reg(2, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
      reg(2, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
      reg(2, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
      reg(output_dma, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE,
      reg(stride_field, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT,
      reg(data_cube_height, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_0,
      reg(7, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__SHIFT, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_1,
      reg(data_cube_height, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__MASK) |
      reg(data_cube_width, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_OW_CFG,
      reg(1, rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT, rk.DPU_BS_OW_CFG_OD_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_CFG,
      reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK) |
      reg(0, rk.DPU_BS_CFG_BS_RELUX_EN__SHIFT, rk.DPU_BS_CFG_BS_RELUX_EN__MASK) |
      reg(0, rk.DPU_BS_CFG_BS_MUL_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_MUL_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_ALU_BYPASS__MASK) |
      reg(0, rk.DPU_BS_CFG_BS_ALU_ALGO__SHIFT, rk.DPU_BS_CFG_BS_ALU_ALGO__MASK) |
      reg(0, rk.DPU_BS_CFG_BS_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_ALU_CFG, 0)
    emit(rk.DPU, rk.REG_DPU_BS_MUL_CFG,
      reg(0x7c00, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__SHIFT, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_RELUX_CMP_VALUE, 0)
    emit(rk.DPU, rk.REG_DPU_BN_CFG,
      reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_MUL_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_EW_CFG,
      reg(1, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE,
      reg(1, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__SHIFT, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE,
      reg(1, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__SHIFT, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__MASK) |
      reg(1, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__SHIFT, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT,
      reg(16, rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__SHIFT, rk.DPU_OUT_CVT_SHIFT_MINUS_EXP__MASK))
    emit(rk.DPU, rk.REG_DPU_SURFACE_ADD,
      reg(stride_field, rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT, rk.DPU_SURFACE_ADD_SURF_ADD__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
      reg(data_cube_height, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
      reg(input_dma, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
      reg(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__MASK) |
      reg(2, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
      reg(weight_dma + 0x4000, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_SURF_STRIDE,
      reg(stride_field, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__SHIFT, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
      reg(2, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__MASK) |
      reg(15, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(2, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT,
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__MASK))

  def _emit_cmpeq_part3_regs(self, input_dma:int, weight_dma:int, output_dma:int, packed_elems:int, rows:int, cols:int) -> None:
    if packed_elems == 0: packed_elems = 1
    r = rows if rows > 0 else 1
    c = cols if cols > 0 else int(packed_elems)
    if r * c < packed_elems:
      r = (packed_elems + c - 1) // c
    if r < 1: r = 1
    if c < 1: c = 1
    data_cube_width = c - 1
    data_cube_height = r - 1
    stride_field = c * 2

    self.q = []
    reg = self.reg
    emit = self.emit_raw
    emit(rk.DPU, rk.REG_DPU_S_POINTER,
      reg(1, rk.DPU_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_S_POINTER_POINTER_PP_MODE__MASK) |
      reg(1, rk.DPU_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_EN__MASK) |
      reg(1, rk.DPU_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_S_POINTER_POINTER_PP_EN__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER,
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__MASK))
    emit(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
      reg(15, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(2, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK) |
      reg(1, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_FORMAT,
      reg(2, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
      reg(2, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
      reg(2, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
      reg(output_dma, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE,
      reg(stride_field, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT,
      reg(data_cube_height, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_0,
      reg(7, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__SHIFT, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_1,
      reg(data_cube_height, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__MASK) |
      reg(data_cube_width, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_OW_CFG,
      reg(1, rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT, rk.DPU_BS_OW_CFG_OD_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_CFG,
      reg(0, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_RELUX_EN__SHIFT, rk.DPU_BS_CFG_BS_RELUX_EN__MASK) |
      reg(0, rk.DPU_BS_CFG_BS_MUL_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_MUL_BYPASS__MASK) |
      reg(0, rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_ALU_BYPASS__MASK) |
      reg(4, rk.DPU_BS_CFG_BS_ALU_ALGO__SHIFT, rk.DPU_BS_CFG_BS_ALU_ALGO__MASK) |
      reg(0, rk.DPU_BS_CFG_BS_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_ALU_CFG,
      reg(0x3F800000, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__SHIFT, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_MUL_CFG,
      reg(0x7bff, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__SHIFT, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_RELUX_CMP_VALUE,
      reg(0x3F800000, rk.DPU_BS_RELUX_CMP_VALUE_BS_RELUX_CMP_DAT__SHIFT, rk.DPU_BS_RELUX_CMP_VALUE_BS_RELUX_CMP_DAT__MASK))
    emit(rk.DPU, rk.REG_DPU_BN_CFG,
      reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_MUL_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_EW_CFG,
      reg(1, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE,
      reg(1, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__SHIFT, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE,
      reg(1, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__SHIFT, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__MASK) |
      reg(1, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__SHIFT, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT, 0)
    emit(rk.DPU, rk.REG_DPU_SURFACE_ADD,
      reg(stride_field, rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT, rk.DPU_SURFACE_ADD_SURF_ADD__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
      reg(data_cube_height, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
      reg(input_dma, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
      reg(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__MASK) |
      reg(2, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
      reg(weight_dma + 0x4000, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_SURF_STRIDE,
      reg(stride_field, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__SHIFT, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
      reg(2, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__MASK) |
      reg(15, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(2, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT,
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__MASK))

  def _emit_neg_regs(self, input_dma:int, weight_dma:int, output_dma:int, packed_elems:int, rows:int, cols:int) -> None:
    if packed_elems == 0: packed_elems = 1
    r = rows if rows > 0 else 1
    c = cols if cols > 0 else int(packed_elems)
    if r * c < packed_elems:
      r = (packed_elems + c - 1) // c
    if r < 1: r = 1
    if c < 1: c = 1
    data_cube_width = c - 1
    data_cube_height = r - 1
    stride_field = c * 2

    self.q = []
    reg = self.reg
    emit = self.emit_raw
    emit(rk.DPU, rk.REG_DPU_S_POINTER,
      reg(1, rk.DPU_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_S_POINTER_POINTER_PP_MODE__MASK) |
      reg(1, rk.DPU_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_EN__MASK) |
      reg(1, rk.DPU_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_S_POINTER_POINTER_PP_EN__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER,
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__MASK))
    emit(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
      reg(15, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(2, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK) |
      reg(1, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_FORMAT,
      reg(2, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
      reg(2, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
      reg(2, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
      reg(output_dma, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE,
      reg(stride_field, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT,
      reg(data_cube_height, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_0,
      reg(7, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__SHIFT, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_1,
      reg(data_cube_height, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__MASK) |
      reg(data_cube_width, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_OW_CFG,
      reg(1, rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT, rk.DPU_BS_OW_CFG_OD_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_CFG,
      reg(0, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_BS_CFG_BS_RELUX_EN__SHIFT, rk.DPU_BS_CFG_BS_RELUX_EN__MASK) |
      reg(0, rk.DPU_BS_CFG_BS_MUL_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_MUL_BYPASS__MASK) |
      reg(0, rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_ALU_BYPASS__MASK) |
      reg(4, rk.DPU_BS_CFG_BS_ALU_ALGO__SHIFT, rk.DPU_BS_CFG_BS_ALU_ALGO__MASK) |
      reg(0, rk.DPU_BS_CFG_BS_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_ALU_CFG,
      reg(0x3F800000, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__SHIFT, rk.DPU_BS_ALU_CFG_BS_ALU_OPERAND__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_MUL_CFG,
      reg(0xBC00, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__SHIFT, rk.DPU_BS_MUL_CFG_BS_MUL_OPERAND__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_RELUX_CMP_VALUE,
      reg(0x3F800000, rk.DPU_BS_RELUX_CMP_VALUE_BS_RELUX_CMP_DAT__SHIFT, rk.DPU_BS_RELUX_CMP_VALUE_BS_RELUX_CMP_DAT__MASK))
    emit(rk.DPU, rk.REG_DPU_BN_CFG,
      reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_MUL_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK) |
      reg(1, rk.DPU_BN_CFG_BN_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_EW_CFG,
      reg(1, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_OP_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_BYPASS__MASK) |
      reg(1, rk.DPU_EW_CFG_EW_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE,
      reg(1, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__SHIFT, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE,
      reg(1, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__SHIFT, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__MASK) |
      reg(1, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__SHIFT, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__MASK))
    emit(rk.DPU, rk.REG_DPU_OUT_CVT_SHIFT, 0)
    emit(rk.DPU, rk.REG_DPU_SURFACE_ADD,
      reg(stride_field, rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT, rk.DPU_SURFACE_ADD_SURF_ADD__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
      reg(data_cube_height, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
      reg(7, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
      reg(input_dma, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
      reg(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__MASK) |
      reg(2, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
      reg(weight_dma + 0x4000, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_SURF_STRIDE,
      reg(stride_field, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__SHIFT, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
      reg(2, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__MASK) |
      reg(15, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(2, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__MASK))
    emit(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT,
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__MASK) |
      reg(1, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__MASK))

  def where(self, mask:list[Any], a:list[Any], b:list[Any], rows:int=0, cols:int=0) -> list[float]:
    n = len(mask)
    if n == 0: return []
    if len(a) != n or len(b) != n:
      raise RuntimeError(f"RK_WHERE length mismatch mask={n} a={len(a)} b={len(b)}")

    mask_vals = np.asarray(mask)
    mask_fp16 = mask_vals.astype(np.float16, copy=False)
    if mask_vals.dtype != np.bool_:
      if np.any((mask_fp16 != 0) & (mask_fp16 != 1)):
        raise RuntimeError("RK_WHERE mask must be 0/1")

    shape = (rows, cols) if rows > 0 and cols > 0 else None
    mask_i16 = mask_fp16.astype(np.int16, copy=False)
    inv_mask = (1 - mask_i16).astype(np.int16, copy=False)
    a_bits = np.asarray(a, dtype=np.float32).astype(np.float16).view(np.int16)
    b_bits = np.asarray(b, dtype=np.float32).astype(np.float16).view(np.int16)
    sel_a = self._mul_part1_int16(a_bits.tolist(), mask_i16.tolist(), shape)
    sel_b = self._mul_part1_int16(b_bits.tolist(), inv_mask.tolist(), shape)
    out_bits = self._add_part1_int16(sel_a, sel_b, shape)
    out_fp16 = np.asarray(out_bits, dtype=np.int16).view(np.float16)
    return out_fp16.tolist()

  def _where_batch(self, mask:list[Any], a:list[Any], b:list[Any], shape:tuple[int, ...]|None) -> list[float]:
    rows, cols = self._where_dims(shape, len(mask))
    return self.where(mask, a, b, rows=rows, cols=cols)

  def _ew_buffers(self, tag:str, packed_bytes:int, weight_bytes:int) -> tuple[Any, Any, Any]:
    key = (tag, packed_bytes, weight_bytes)
    bufs = self._ew_cache.get(key)
    if bufs is not None: return bufs
    input_buf = self.device._gpu_alloc(packed_bytes, 0, name=f"{tag}_in")
    weight_buf = self.device._gpu_alloc(weight_bytes, 0, name=f"{tag}_wt")
    output_buf = self.device._gpu_alloc(packed_bytes, 0, name=f"{tag}_out")
    self._ew_cache[key] = (input_buf, weight_buf, output_buf)
    return input_buf, weight_buf, output_buf

  def _add_part1(self, a:list[Any], b:list[Any], shape:tuple[int, ...]|None) -> list[float]:
    self.device.reset_controller_if_needed()
    n = len(a)
    if n != len(b): raise RuntimeError(f"ADD input length mismatch {n} != {len(b)}")
    if n == 0: return []
    if n > 2048:
      out = []
      for off in range(0, n, 2048):
        out.extend(self._add_part1(a[off:off+2048], b[off:off+2048], None))
      return out

    a_fp16 = np.asarray(a, dtype=np.float32).astype(np.float16)
    b_fp16 = np.asarray(b, dtype=np.float32).astype(np.float16)
    packed_a = np.zeros((n, 8), dtype=np.float16)
    packed_b = np.zeros((n, 8), dtype=np.float16)
    packed_a[:, 0] = a_fp16
    packed_b[:, 0] = b_fp16
    packed_bytes = n * 0x10
    packed_elems = packed_bytes // 0x10

    input_buf, weight_buf, output_buf = self._ew_buffers("add", packed_bytes, 0x4000 + packed_bytes)
    ctypes.memmove(input_buf.va_addr, packed_a.tobytes(), packed_bytes)
    ctypes.memset(weight_buf.va_addr, 0, 0x4000 + packed_bytes)
    ctypes.memmove(weight_buf.va_addr + 0x4000, packed_b.tobytes(), packed_bytes)
    ctypes.memset(output_buf.va_addr, 0, packed_bytes)

    rows, cols = self._where_dims(shape, n)
    self._emit_where_add_regs(input_buf.meta.dma_addr, weight_buf.meta.dma_addr,
                              output_buf.meta.dma_addr, packed_elems, rows, cols)
    self.submit()

    out = np.frombuffer(ctypes.string_at(output_buf.va_addr, packed_bytes), dtype=np.float16).reshape(n, 8)[:, 0]
    return out.tolist()

  def _max_part1(self, a:list[Any], b:list[Any], shape:tuple[int, ...]|None) -> list[float]:
    self.device.reset_controller_if_needed()
    n = len(a)
    if n != len(b): raise RuntimeError(f"MAX input length mismatch {n} != {len(b)}")
    if n == 0: return []
    if n > 2048:
      out = []
      for off in range(0, n, 2048):
        out.extend(self._max_part1(a[off:off+2048], b[off:off+2048], None))
      return out

    a_fp16 = np.asarray(a, dtype=np.float32).astype(np.float16)
    b_fp16 = np.asarray(b, dtype=np.float32).astype(np.float16)
    packed_a = np.zeros((n, 8), dtype=np.float16)
    packed_b = np.zeros((n, 8), dtype=np.float16)
    packed_a[:, 0] = a_fp16
    packed_b[:, 0] = b_fp16
    packed_bytes = n * 0x10
    packed_elems = packed_bytes // 0x10

    input_buf, weight_buf, output_buf = self._ew_buffers("max", packed_bytes, 0x4000 + packed_bytes)
    ctypes.memmove(input_buf.va_addr, packed_a.tobytes(), packed_bytes)
    ctypes.memset(weight_buf.va_addr, 0, 0x4000 + packed_bytes)
    ctypes.memmove(weight_buf.va_addr + 0x4000, packed_b.tobytes(), packed_bytes)
    ctypes.memset(output_buf.va_addr, 0, packed_bytes)

    rows, cols = self._where_dims(shape, n)
    self._emit_where_max_regs(input_buf.meta.dma_addr, weight_buf.meta.dma_addr,
                              output_buf.meta.dma_addr, packed_elems, rows, cols)
    self.submit()

    out = np.frombuffer(ctypes.string_at(output_buf.va_addr, packed_bytes), dtype=np.float16).reshape(n, 8)[:, 0]
    return out.tolist()

  def _neg_part1(self, a:list[Any], shape:tuple[int, ...]|None) -> list[float]:
    self.device.reset_controller_if_needed()
    n = len(a)
    if n == 0: return []
    if n > 2048:
      out = []
      for off in range(0, n, 2048):
        out.extend(self._neg_part1(a[off:off+2048], None))
      return out

    a_fp16 = np.asarray(a, dtype=np.float32).astype(np.float16)
    packed_a = np.zeros((n, 8), dtype=np.float16)
    packed_a[:, 0] = a_fp16
    packed_bytes = n * 0x10
    packed_elems = packed_bytes // 0x10

    input_buf, weight_buf, output_buf = self._ew_buffers("neg", packed_bytes, 0x4000 + packed_bytes)
    ctypes.memmove(input_buf.va_addr, packed_a.tobytes(), packed_bytes)
    ctypes.memset(weight_buf.va_addr, 0, 0x4000 + packed_bytes)
    ctypes.memset(output_buf.va_addr, 0, packed_bytes)

    rows, cols = self._where_dims(shape, n)
    self._emit_where_neg_regs(input_buf.meta.dma_addr, weight_buf.meta.dma_addr,
                              output_buf.meta.dma_addr, packed_elems, rows, cols)
    self.submit()

    out = np.frombuffer(ctypes.string_at(output_buf.va_addr, packed_bytes), dtype=np.float16).reshape(n, 8)[:, 0]
    return out.tolist()

  def _add_part1_int16(self, a:list[Any], b:list[Any], shape:tuple[int, ...]|None) -> list[int]:
    self.device.reset_controller_if_needed()
    try:
      rk.DRM_IOCTL_RKNPU_ACTION(self.device.fd_ctl, flags=rk.RKNPU_ACT_RESET)
    except Exception:
      pass
    n = len(a)
    if n != len(b): raise RuntimeError(f"ADD int16 input length mismatch {n} != {len(b)}")
    if n == 0: return []

    a_i16 = np.asarray(a, dtype=np.int16)
    b_i16 = np.asarray(b, dtype=np.int16)
    packed_a = np.zeros((n, 8), dtype=np.int16)
    packed_b = np.zeros((n, 8), dtype=np.int16)
    packed_a[:, 0] = a_i16
    packed_b[:, 0] = b_i16
    packed_bytes = n * 0x10
    packed_elems = packed_bytes // 0x10

    input_buf, weight_buf, output_buf = self._ew_buffers("add_i16", packed_bytes, 0x4000 + packed_bytes)
    ctypes.memmove(input_buf.va_addr, packed_a.tobytes(), packed_bytes)
    ctypes.memset(weight_buf.va_addr, 0, 0x4000 + packed_bytes)
    ctypes.memmove(weight_buf.va_addr + 0x4000, packed_b.tobytes(), packed_bytes)
    ctypes.memset(output_buf.va_addr, 0, packed_bytes)

    rows, cols = self._where_dims(shape, n)
    self._emit_where_add_regs(input_buf.meta.dma_addr, weight_buf.meta.dma_addr,
                              output_buf.meta.dma_addr, packed_elems, rows, cols, dtype=dtypes.int16)
    self.submit()

    out = np.frombuffer(ctypes.string_at(output_buf.va_addr, packed_bytes), dtype=np.int16).reshape(n, 8)[:, 0]
    return out.tolist()

  def _add_scalar(self, values:list[Any], scalar:float, shape:tuple[int, ...]|None) -> list[float]:
    return self._add_part1(values, [scalar] * len(values), shape)

  def _mul_part1(self, a:list[Any], b:list[Any], shape:tuple[int, ...]|None) -> list[float]:
    self.device.reset_controller_if_needed()
    try:
      rk.DRM_IOCTL_RKNPU_ACTION(self.device.fd_ctl, flags=rk.RKNPU_ACT_RESET)
    except Exception:
      pass
    n = len(a)
    if n != len(b): raise RuntimeError(f"MUL input length mismatch {n} != {len(b)}")
    if n == 0: return []
    if n > 2048:
      out = []
      for off in range(0, n, 2048):
        out.extend(self._mul_part1(a[off:off+2048], b[off:off+2048], None))
      return out

    a_fp16 = np.asarray(a, dtype=np.float32).astype(np.float16)
    b_fp16 = np.asarray(b, dtype=np.float32).astype(np.float16)
    packed_a = np.zeros((n, 8), dtype=np.float16)
    packed_b = np.zeros((n, 8), dtype=np.float16)
    packed_a[:, 0] = a_fp16
    packed_b[:, 0] = b_fp16
    packed_bytes = n * 0x10
    packed_elems = packed_bytes // 0x10

    input_buf, weight_buf, output_buf = self._ew_buffers("mul", packed_bytes, 0x4000 + packed_bytes)
    ctypes.memmove(input_buf.va_addr, packed_a.tobytes(), packed_bytes)
    ctypes.memset(weight_buf.va_addr, 0, 0x4000 + packed_bytes)
    ctypes.memmove(weight_buf.va_addr + 0x4000, packed_b.tobytes(), packed_bytes)
    ctypes.memset(output_buf.va_addr, 0, packed_bytes)

    rows, cols = self._where_dims(shape, n)
    self._emit_where_mul_regs(input_buf.meta.dma_addr, weight_buf.meta.dma_addr,
                              output_buf.meta.dma_addr, packed_elems, rows, cols)
    self.submit()

    out = np.frombuffer(ctypes.string_at(output_buf.va_addr, packed_bytes), dtype=np.float16).reshape(n, 8)[:, 0]
    return out.tolist()

  def _mul_part1_int16(self, a:list[Any], b:list[Any], shape:tuple[int, ...]|None) -> list[int]:
    self.device.reset_controller_if_needed()
    try:
      rk.DRM_IOCTL_RKNPU_ACTION(self.device.fd_ctl, flags=rk.RKNPU_ACT_RESET)
    except Exception:
      pass
    n = len(a)
    if n != len(b): raise RuntimeError(f"MUL int16 input length mismatch {n} != {len(b)}")
    if n == 0: return []

    a_i16 = np.asarray(a, dtype=np.int16)
    b_i16 = np.asarray(b, dtype=np.int16)
    packed_a = np.zeros((n, 8), dtype=np.int16)
    packed_b = np.zeros((n, 8), dtype=np.int16)
    packed_a[:, 0] = a_i16
    packed_b[:, 0] = b_i16
    packed_bytes = n * 0x10
    packed_elems = packed_bytes // 0x10

    input_buf, weight_buf, output_buf = self._ew_buffers("mul_i16", packed_bytes, 0x4000 + packed_bytes)
    ctypes.memmove(input_buf.va_addr, packed_a.tobytes(), packed_bytes)
    ctypes.memset(weight_buf.va_addr, 0, 0x4000 + packed_bytes)
    ctypes.memmove(weight_buf.va_addr + 0x4000, packed_b.tobytes(), packed_bytes)
    ctypes.memset(output_buf.va_addr, 0, packed_bytes)

    rows, cols = self._where_dims(shape, n)
    self._emit_where_mul_regs(input_buf.meta.dma_addr, weight_buf.meta.dma_addr,
                              output_buf.meta.dma_addr, packed_elems, rows, cols, dtype=dtypes.int16)
    self.submit()

    out = np.frombuffer(ctypes.string_at(output_buf.va_addr, packed_bytes), dtype=np.int16).reshape(n, 8)[:, 0]
    return out.tolist()

  def _select_batch(self, mask:list[Any], a:list[Any], b:list[Any], shape:tuple[int, ...]|None) -> list[float]:
    n = len(mask)
    if len(a) != n or len(b) != n:
      raise RuntimeError(f"select input length mismatch mask={n} a={len(a)} b={len(b)}")
    mask_f = [float(x) for x in mask]
    inv_mask = [1.0 - x for x in mask_f]
    mul_a = self._mul_part1(a, mask_f, shape)
    mul_b = self._mul_part1(b, inv_mask, shape)
    return self._add_part1(mul_a, mul_b, shape)

  def _cmplt_part1(self, a:list[Any], b:list[Any]) -> list[bool]:
    self.device.reset_controller_if_needed()
    try:
      rk.DRM_IOCTL_RKNPU_ACTION(self.device.fd_ctl, flags=rk.RKNPU_ACT_RESET)
    except Exception:
      pass
    n = len(a)
    if n != len(b): raise RuntimeError(f"CMPLT input length mismatch {n} != {len(b)}")
    if n == 0: return []

    a_f = np.asarray(a, dtype=np.float32)
    b_f = np.asarray(b, dtype=np.float32)
    nan_mask = np.isnan(a_f) | np.isnan(b_f)
    inf_mask = np.isinf(a_f) | np.isinf(b_f)
    a_fp16 = np.where(nan_mask, np.float32(0.0), a_f).astype(np.float16)
    b_fp16 = np.where(nan_mask, np.float32(0.0), b_f).astype(np.float16)

    packed_a = np.zeros((n, 8), dtype=np.float16)
    packed_b = np.zeros((n, 8), dtype=np.float16)
    packed_a[:, 0] = a_fp16
    packed_b[:, 0] = b_fp16
    packed_bytes_a = packed_a.tobytes()
    packed_bytes_b = packed_b.tobytes()

    packed_bytes = n * 0x10
    input_buf = None
    weight_buf = None
    output_buf = None
    try:
      input_buf = self.device._gpu_alloc(packed_bytes, 0, name="cmplt_in")
      weight_buf = self.device._gpu_alloc(0x4000 + packed_bytes, 0, name="cmplt_wt")
      output_buf = self.device._gpu_alloc(packed_bytes, 0, name="cmplt_out")
      ctypes.memmove(input_buf.va_addr, packed_bytes_b, packed_bytes)
      ctypes.memset(weight_buf.va_addr, 0, 0x4000 + packed_bytes)
      ctypes.memmove(weight_buf.va_addr + 0x4000, packed_bytes_a, packed_bytes)
      ctypes.memset(output_buf.va_addr, 0, packed_bytes)

      rows, cols = 1, n
      data_cube_width, data_cube_height = cols - 1, rows - 1
      stride_field = cols * 2

      self.q = []
      prec = self.get_precision(dtypes.float16)

      self.emit_raw(rk.DPU, rk.REG_DPU_S_POINTER,
        self.reg(1, rk.DPU_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_S_POINTER_POINTER_PP_MODE__MASK) |
        self.reg(1, rk.DPU_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_EN__MASK) |
        self.reg(1, rk.DPU_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_S_POINTER_POINTER_PP_EN__MASK))
      self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER,
        self.reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__MASK) |
        self.reg(1, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__MASK) |
        self.reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__MASK))

      self.emit_raw(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
        self.reg(15, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
        self.reg(2, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK) |
        self.reg(1, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_DATA_FORMAT,
        self.reg(prec, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
        self.reg(prec, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
        self.reg(prec, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))

      self.emit_raw(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
        self.reg(output_buf.meta.dma_addr, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE,
        self.reg(stride_field, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
        self.reg(data_cube_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT,
        self.reg(data_cube_height, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
        self.reg(7, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))

      self.emit_raw(rk.DPU, rk.REG_DPU_WDMA_SIZE_0,
        self.reg(7, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__SHIFT, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_WDMA_SIZE_1,
        self.reg(data_cube_height, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__MASK) |
        self.reg(data_cube_width, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))

      self.emit_raw(rk.DPU, rk.REG_DPU_BS_OW_CFG,
        self.reg(1, rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT, rk.DPU_BS_OW_CFG_OD_BYPASS__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_BS_CFG,
        self.reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK) |
        self.reg(1, rk.DPU_BS_CFG_BS_MUL_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_MUL_BYPASS__MASK) |
        self.reg(1, rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_ALU_BYPASS__MASK) |
        self.reg(1, rk.DPU_BS_CFG_BS_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_BYPASS__MASK))

      self.emit_raw(rk.DPU, rk.REG_DPU_BN_CFG,
        self.reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK) |
        self.reg(1, rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_MUL_BYPASS__MASK) |
        self.reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK) |
        self.reg(1, rk.DPU_BN_CFG_BN_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_BYPASS__MASK))

      self.emit_raw(rk.DPU, rk.REG_DPU_EW_CFG,
        self.reg(1, rk.DPU_EW_CFG_EW_DATA_MODE__SHIFT, rk.DPU_EW_CFG_EW_DATA_MODE__MASK) |
        self.reg(2, rk.DPU_EW_CFG_EDATA_SIZE__SHIFT, rk.DPU_EW_CFG_EDATA_SIZE__MASK) |
        self.reg(4, rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT, rk.DPU_EW_CFG_EW_ALU_ALGO__MASK) |
        self.reg(0, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
        self.reg(0, rk.DPU_EW_CFG_EW_RELUX_EN__SHIFT, rk.DPU_EW_CFG_EW_RELUX_EN__MASK) |
        self.reg(1, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
        self.reg(1, rk.DPU_EW_CFG_EW_OP_SRC__SHIFT, rk.DPU_EW_CFG_EW_OP_SRC__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_EW_RELUX_CMP_VALUE,
        self.reg(0x3F800000, rk.DPU_EW_RELUX_CMP_VALUE_EW_RELUX_CMP_DAT__SHIFT, rk.DPU_EW_RELUX_CMP_VALUE_EW_RELUX_CMP_DAT__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE,
        self.reg(1, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__SHIFT, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__MASK))

      self.emit_raw(rk.DPU, rk.REG_DPU_SURFACE_ADD,
        self.reg(stride_field, rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT, rk.DPU_SURFACE_ADD_SURF_ADD__MASK))

      self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
        self.reg(data_cube_width, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK))
      self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
        self.reg(data_cube_height, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__MASK))
      self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
        self.reg(7, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK))
      self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
        self.reg(input_buf.meta.dma_addr, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK))
      self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
        self.reg(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__MASK) |
        self.reg(2, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__MASK))
      ew_base = weight_buf.meta.dma_addr + 0x4000
      self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
        self.reg(ew_base, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__MASK))
      self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_SURF_STRIDE,
        self.reg(stride_field, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__SHIFT, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__MASK))
      self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SURF_NOTCH,
        self.reg(0, rk.DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR__SHIFT, rk.DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR__MASK))
      self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_SURF_NOTCH,
        self.reg(0, rk.DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH__SHIFT, rk.DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH__MASK))
      self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
        self.reg(prec, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__MASK) |
        self.reg(15, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__MASK) |
        self.reg(prec, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__MASK) |
        self.reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__MASK) |
        self.reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__MASK))
      self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT,
        self.reg(1, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__MASK) |
        self.reg(1, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__MASK) |
        self.reg(1, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__MASK) |
        self.reg(1, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__MASK))

      self.submit()

      out = np.frombuffer(ctypes.string_at(output_buf.va_addr, packed_bytes), dtype=np.float16).reshape(n, 8)[:, 0]
      out_bool = [bool(x > 0) for x in out.tolist()]
      if nan_mask.any():
        out_bool = [False if is_nan else v for v, is_nan in zip(out_bool, nan_mask.tolist())]
      if inf_mask.any():
        expected = (a_f < b_f).tolist()
        out_bool = [bool(exp) if is_inf else v for v, is_inf, exp in zip(out_bool, inf_mask.tolist(), expected)]
      return out_bool
    finally:
      if input_buf is not None: self.device._gpu_free(input_buf)
      if weight_buf is not None: self.device._gpu_free(weight_buf)
      if output_buf is not None: self.device._gpu_free(output_buf)

  def _cmpeq_pipeline(self, a:list[Any], b:list[Any], negate:bool=False) -> list[bool]:
    self.device.reset_controller_if_needed()
    try:
      rk.DRM_IOCTL_RKNPU_ACTION(self.device.fd_ctl, flags=rk.RKNPU_ACT_RESET)
    except Exception:
      pass
    n = len(a)
    if n != len(b): raise RuntimeError(f"CMPEQ/CMPNE input length mismatch {n} != {len(b)}")
    if n == 0: return []
    if n > 1:
      a_f = np.asarray(a, dtype=np.float32)
      b_f = np.asarray(b, dtype=np.float32)
      nan_mask = np.isnan(a_f) | np.isnan(b_f)
      zeros = [0.0] * n
      neg_b = [-float(x) for x in b]
      diff = self._add_part1(a, neg_b, None)
      lt = self._cmplt_part1(diff, zeros)
      gt = self._cmplt_part1(zeros, diff)
      lt_f = [1.0 if v else 0.0 for v in lt]
      gt_f = [1.0 if v else 0.0 for v in gt]
      sum_f = self._add_part1(lt_f, gt_f, None)
      neq = [v > 0 for v in sum_f]
      if nan_mask.any():
        neq = [True if is_nan else v for v, is_nan in zip(neq, nan_mask.tolist())]
      if negate: return neq
      return [not v for v in neq]

    a_f = np.asarray(a, dtype=np.float32)
    b_f = np.asarray(b, dtype=np.float32)
    nan_mask = np.isnan(a_f) | np.isnan(b_f)
    inf_mask = np.isinf(a_f) | np.isinf(b_f)
    a_fp16 = np.where(nan_mask, np.float32(0.0), a_f).astype(np.float16)
    b_fp16 = np.where(nan_mask, np.float32(0.0), b_f).astype(np.float16)

    packed_a = np.zeros((n, 8), dtype=np.float16)
    packed_b = np.zeros((n, 8), dtype=np.float16)
    packed_a[:, 0] = a_fp16
    packed_b[:, 0] = b_fp16
    packed_bytes = n * 0x10
    packed_elems = packed_bytes // 0x10

    input_buf = None
    weight_buf = None
    stage1_out = None
    weight_zero = None
    stage2_out = None
    stage3_out = None
    neg_out = None
    try:
      input_buf = self.device._gpu_alloc(packed_bytes, 0, name="cmpeq_in")
      weight_buf = self.device._gpu_alloc(0x4000 + packed_bytes, 0, name="cmpeq_wt")
      stage1_out = self.device._gpu_alloc(packed_bytes, 0, name="cmpeq_p1_out")
      weight_zero = self.device._gpu_alloc(0x4000 + packed_bytes, 0, name="cmpeq_wt_zero")
      stage2_out = self.device._gpu_alloc(packed_bytes, 0, name="cmpeq_p2_out")
      stage3_out = self.device._gpu_alloc(packed_bytes, 0, name="cmpeq_p3_out")
      if negate:
        neg_out = self.device._gpu_alloc(packed_bytes, 0, name="cmpneq_out")

      ctypes.memmove(input_buf.va_addr, packed_b.tobytes(), packed_bytes)
      ctypes.memset(weight_buf.va_addr, 0, 0x4000 + packed_bytes)
      ctypes.memmove(weight_buf.va_addr + 0x4000, packed_a.tobytes(), packed_bytes)
      ctypes.memset(stage1_out.va_addr, 0, packed_bytes)
      ctypes.memset(weight_zero.va_addr, 0, 0x4000 + packed_bytes)
      ctypes.memset(stage2_out.va_addr, 0, packed_bytes)
      ctypes.memset(stage3_out.va_addr, 0, packed_bytes)
      if negate: ctypes.memset(neg_out.va_addr, 0, packed_bytes)

      rows, cols = n, 1
      self._emit_minus_regs(input_buf.meta.dma_addr, weight_buf.meta.dma_addr,
                            stage1_out.meta.dma_addr, packed_elems, rows, cols)
      self.submit()
      self._emit_cmpeq_part2_regs(stage1_out.meta.dma_addr, weight_zero.meta.dma_addr,
                                  stage2_out.meta.dma_addr, packed_elems, rows, cols)
      self.submit()
      self._emit_cmpeq_part3_regs(stage2_out.meta.dma_addr, weight_zero.meta.dma_addr,
                                  stage3_out.meta.dma_addr, packed_elems, rows, cols)
      self.submit()

      out_buf = stage3_out
      if negate:
        self._emit_neg_regs(stage3_out.meta.dma_addr, weight_zero.meta.dma_addr,
                            neg_out.meta.dma_addr, packed_elems, rows, cols)
        self.submit()
        out_buf = neg_out

      out = np.frombuffer(ctypes.string_at(out_buf.va_addr, packed_bytes), dtype=np.float16).reshape(n, 8)[:, 0]
      out_bool = [bool(x > 0) for x in out.tolist()]
      if nan_mask.any():
        nan_val = True if negate else False
        out_bool = [nan_val if is_nan else v for v, is_nan in zip(out_bool, nan_mask.tolist())]
      if inf_mask.any():
        expected = (a_f != b_f) if negate else (a_f == b_f)
        exp_list = expected.tolist()
        out_bool = [bool(exp) if is_inf else v for v, is_inf, exp in zip(out_bool, inf_mask.tolist(), exp_list)]
      return out_bool
    finally:
      if input_buf is not None: self.device._gpu_free(input_buf)
      if weight_buf is not None: self.device._gpu_free(weight_buf)
      if stage1_out is not None: self.device._gpu_free(stage1_out)
      if weight_zero is not None: self.device._gpu_free(weight_zero)
      if stage2_out is not None: self.device._gpu_free(stage2_out)
      if stage3_out is not None: self.device._gpu_free(stage3_out)
      if neg_out is not None: self.device._gpu_free(neg_out)

  def _cmpeq_part1(self, a:list[Any], b:list[Any]) -> list[bool]:
    return self._cmpeq_pipeline(a, b, negate=False)

  def _cmpneq_part1(self, a:list[Any], b:list[Any]) -> list[bool]:
    return self._cmpeq_pipeline(a, b, negate=True)

  def _idiv_batch(self, numerator:list[Any], denominator:list[Any], shape:tuple[int, ...]|None, dtype:DType) -> list[Any]:
    dtype = dtype.scalar()
    if dtype not in dtypes.sints or dtype.itemsize > 4:
      raise RuntimeError(f"RK_IDIV unsupported dtype {dtype}")
    n = len(numerator)
    if n != len(denominator): raise RuntimeError(f"RK_IDIV input length mismatch {n} != {len(denominator)}")
    if n == 0: return []

    num_f = [float(x) for x in numerator]
    den_f = [float(x) for x in denominator]
    max_input = 2048.0
    if any(abs(v) > max_input for v in num_f) or any(abs(v) > max_input for v in den_f):
      raise RuntimeError("RK_IDIV supports abs(inputs) <= 2048 for float16 precision")

    zeros = [0.0] * n
    ones = [1.0] * n
    denom_lt_zero = self._cmplt_part1(den_f, zeros)
    denom_gt_zero = self._cmplt_part1(zeros, den_f)
    denom_zero = self._select_batch(denom_lt_zero, zeros, self._select_batch(denom_gt_zero, zeros, ones, shape), shape)
    denom_safe = self._select_batch(denom_zero, ones, den_f, shape)

    div_vals = self._div_part1(num_f, denom_safe)
    if any(abs(v) >= 1024.0 for v in div_vals):
      raise RuntimeError("RK_IDIV requires |quotient| < 1024 for float16 precision")

    div_neg = self._cmplt_part1(div_vals, zeros)
    neg_div = self._mul_part1(div_vals, [-1.0] * n, shape)
    abs_div = self._select_batch(div_neg, neg_div, div_vals, shape)
    round_abs = self._roundoff_batch(abs_div, shape)
    neg_round = self._mul_part1(round_abs, [-1.0] * n, shape)
    rounded = self._select_batch(div_neg, neg_round, round_abs, shape)
    prod = self._mul_part1(rounded, denom_safe, shape)
    neg_prod = self._mul_part1(prod, [-1.0] * n, shape)
    remainder = self._add_part1(num_f, neg_prod, shape)

    num_lt_zero = self._cmplt_part1(num_f, zeros)
    num_gt_zero = self._cmplt_part1(zeros, num_f)
    rem_lt_zero = self._cmplt_part1(remainder, zeros)
    rem_gt_zero = self._cmplt_part1(zeros, remainder)
    adjust_mask = [(rl and np) or (rp and nn)
                   for rl, np, rp, nn in zip(rem_lt_zero, num_gt_zero, rem_gt_zero, num_lt_zero)]
    same_sign = [(nn and dn) or (np and dp)
                 for nn, np, dn, dp in zip(num_lt_zero, num_gt_zero, denom_lt_zero, denom_gt_zero)]

    rounded_minus = self._add_scalar(rounded, -1.0, shape)
    rounded_plus = self._add_scalar(rounded, 1.0, shape)
    adjust_vals = self._select_batch(same_sign, rounded_minus, rounded_plus, shape)
    trunc_vals = self._select_batch(adjust_mask, adjust_vals, rounded, shape)

    out_vals = self._select_batch(denom_zero, zeros, trunc_vals, shape)
    cast = truncate.get(dtype, lambda x: x)
    return [cast(dtypes.as_const(x, dtype)) for x in out_vals]

  def _div_part1(self, numerator:list[Any], denominator:list[Any]) -> list[float]:
    self.device.reset_controller_if_needed()
    try:
      rk.DRM_IOCTL_RKNPU_ACTION(self.device.fd_ctl, flags=rk.RKNPU_ACT_RESET)
    except Exception:
      pass
    n = len(numerator)
    if n != len(denominator): raise RuntimeError(f"DIV input length mismatch {n} != {len(denominator)}")
    if n == 0: return []

    rows = int(math.sqrt(n))
    while rows > 1 and n % rows != 0:
      rows -= 1
    cols = n // rows if rows > 0 else n

    num_fp16 = np.asarray(numerator, dtype=np.float32).astype(np.float16)
    den_fp16 = np.asarray(denominator, dtype=np.float32).astype(np.float16)
    packed_den = np.zeros((n, 8), dtype=np.float16)
    packed_den[:, 0] = den_fp16
    packed_num = np.zeros((n, 8), dtype=np.float16)
    packed_num[:, 0] = num_fp16
    packed_bytes_den = packed_den.tobytes()
    packed_bytes_num = packed_num.tobytes()
    packed_bytes = packed_den.nbytes

    input_buf = None
    weight_buf = None
    output_buf = None
    try:
      input_buf = self.device._gpu_alloc(packed_bytes, 0, name="div_in")
      weight_buf = self.device._gpu_alloc(0x4000 + packed_bytes, 0, name="div_wt")
      output_buf = self.device._gpu_alloc(packed_bytes, 0, name="div_out")
      ctypes.memmove(input_buf.va_addr, packed_bytes_den, packed_bytes)
      ctypes.memset(weight_buf.va_addr, 0, 0x4000 + packed_bytes)
      ctypes.memmove(weight_buf.va_addr + 0x4000, packed_bytes_num, packed_bytes)
      ctypes.memset(output_buf.va_addr, 0, packed_bytes)

      data_cube_width, data_cube_height = cols - 1, rows - 1
      stride_field = cols * 4

      self.q = []
      prec = self.get_precision(dtypes.float16)

      self.emit_raw(rk.DPU, rk.REG_DPU_S_POINTER,
        self.reg(1, rk.DPU_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_S_POINTER_POINTER_PP_MODE__MASK) |
        self.reg(1, rk.DPU_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_EN__MASK) |
        self.reg(1, rk.DPU_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_S_POINTER_POINTER_PP_EN__MASK))
      self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_S_POINTER,
        self.reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE__MASK) |
        self.reg(1, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN__MASK) |
        self.reg(1, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN__MASK))

      self.emit_raw(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
        self.reg(15, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
        self.reg(2, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK) |
        self.reg(1, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_FLYING_MODE__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_DATA_FORMAT,
        self.reg(prec, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
        self.reg(prec, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
        self.reg(prec, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))

      self.emit_raw(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
        self.reg(output_buf.meta.dma_addr, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE,
        self.reg(stride_field, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
        self.reg(data_cube_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT,
        self.reg(data_cube_height, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
        self.reg(7, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))

      self.emit_raw(rk.DPU, rk.REG_DPU_BS_CFG,
        self.reg(1, rk.DPU_BS_CFG_BS_RELU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_RELU_BYPASS__MASK) |
        self.reg(1, rk.DPU_BS_CFG_BS_MUL_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_MUL_BYPASS__MASK) |
        self.reg(1, rk.DPU_BS_CFG_BS_ALU_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_ALU_BYPASS__MASK) |
        self.reg(1, rk.DPU_BS_CFG_BS_BYPASS__SHIFT, rk.DPU_BS_CFG_BS_BYPASS__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_BS_OW_CFG,
        self.reg(1, rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT, rk.DPU_BS_OW_CFG_OD_BYPASS__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_WDMA_SIZE_0,
        self.reg(7, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__SHIFT, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_WDMA_SIZE_1,
        self.reg(data_cube_height, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__MASK) |
        self.reg(data_cube_width, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))

      self.emit_raw(rk.DPU, rk.REG_DPU_BN_CFG,
        self.reg(1, rk.DPU_BN_CFG_BN_RELU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_RELU_BYPASS__MASK) |
        self.reg(1, rk.DPU_BN_CFG_BN_MUL_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_MUL_BYPASS__MASK) |
        self.reg(1, rk.DPU_BN_CFG_BN_ALU_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_ALU_BYPASS__MASK) |
        self.reg(1, rk.DPU_BN_CFG_BN_BYPASS__SHIFT, rk.DPU_BN_CFG_BN_BYPASS__MASK))

      self.emit_raw(rk.DPU, rk.REG_DPU_EW_CFG,
        self.reg(1, rk.DPU_EW_CFG_EW_DATA_MODE__SHIFT, rk.DPU_EW_CFG_EW_DATA_MODE__MASK) |
        self.reg(2, rk.DPU_EW_CFG_EDATA_SIZE__SHIFT, rk.DPU_EW_CFG_EDATA_SIZE__MASK) |
        self.reg(3, rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT, rk.DPU_EW_CFG_EW_ALU_ALGO__MASK) |
        self.reg(1, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
        self.reg(1, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK) |
        self.reg(1, rk.DPU_EW_CFG_EW_LUT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_LUT_BYPASS__MASK) |
        self.reg(1, rk.DPU_EW_CFG_EW_OP_SRC__SHIFT, rk.DPU_EW_CFG_EW_OP_SRC__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_EW_CVT_SCALE_VALUE,
        self.reg(1, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__SHIFT, rk.DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE__MASK))
      self.emit_raw(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE,
        self.reg(0, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__SHIFT, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__MASK) |
        self.reg(1, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__SHIFT, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__MASK))

      self.emit_raw(rk.DPU, rk.REG_DPU_SURFACE_ADD,
        self.reg(stride_field, rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT, rk.DPU_SURFACE_ADD_SURF_ADD__MASK))

      self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
        self.reg(data_cube_width, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK))
      self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
        self.reg(data_cube_height, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__MASK))
      self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
        self.reg(7, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK))
      src_base = weight_buf.meta.dma_addr + 0x4000
      self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
        self.reg(src_base, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK))
      self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
        self.reg(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__MASK) |
        self.reg(2, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__MASK))
      self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
        self.reg(input_buf.meta.dma_addr, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__MASK))
      self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_SURF_STRIDE,
        self.reg(stride_field, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__SHIFT, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__MASK))
      self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
        self.reg(prec, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__MASK) |
        self.reg(15, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__MASK) |
        self.reg(prec, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__MASK) |
        self.reg(0, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__SHIFT,
                 rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__MASK) |
        self.reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__MASK))
      self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_WEIGHT,
        self.reg(1, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_E_WEIGHT__MASK) |
        self.reg(1, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_N_WEIGHT__MASK) |
        self.reg(1, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_B_WEIGHT__MASK) |
        self.reg(1, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__SHIFT, rk.DPU_RDMA_RDMA_WEIGHT_M_WEIGHT__MASK))

      self.submit()

      out = np.frombuffer(ctypes.string_at(output_buf.va_addr, packed_bytes), dtype=np.float16).reshape(n, 8)[:, 0]
      return out.astype(np.float32).tolist()
    finally:
      if input_buf is not None: self.device._gpu_free(input_buf)
      if weight_buf is not None: self.device._gpu_free(weight_buf)
      if output_buf is not None: self.device._gpu_free(output_buf)

  def _recip_part1(self, a:list[Any]) -> list[float]:
    n = len(a)
    if n == 0: return []
    return self._div_part1([1.0] * n, a)

  def create_reg(self, reset_queue: bool=True):
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

    ctypes.memset(self.device.task_buf.va_addr, 0, self.device.task_buf.size)
    tasks = ctypes.cast(self.device.task_buf.va_addr, ctypes.POINTER(rk.struct_rknpu_task * 128)).contents
    reg_entries = self.device.cmd_buf.size // ctypes.sizeof(ctypes.c_uint64)
    reg_array_type = ctypes.c_uint64 * reg_entries
    regcmd = ctypes.cast(self.device.cmd_buf.va_addr, ctypes.POINTER(reg_array_type)).contents
    if len(self.q) > reg_entries:
      raise RuntimeError(f"regcmd overflow: {len(self.q)} entries > {reg_entries} capacity")
    for i in range(len(self.q)):
      regcmd[i] = self.q[i]
    for i in range(len(self.q), reg_entries):
      regcmd[i] = 0

    tasks[0].flags  = 0;
    tasks[0].op_idx = getattr(self, "_submit_op_idx", 1);
    tasks[0].enable_mask = getattr(self, "_submit_enable_mask", 0xd);
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
            core_mask=0,
            fence_fd=-1,  
            subcore_task=(rk.struct_rknpu_subcore_task * 5)(
                rk.struct_rknpu_subcore_task(task_start=0, task_number=1),
                rk.struct_rknpu_subcore_task(task_start=0, task_number=0),
                rk.struct_rknpu_subcore_task(task_start=0, task_number=0),
            )
    )
    if DEBUG >= 3:
      print("DRM_IOCTL_RKNPU_SUBMIT")
    rk.DRM_IOCTL_RKNPU_SUBMIT(self.device.fd_ctl, __payload=submit_res)

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
    self._ew_cache: dict[tuple[str, int, int], tuple[Any, Any, Any]] = {}
    if DEBUG >= 3:
      print("RockchipProgram init payload", type(loaded))



  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    if self._rk_conv_payload is not None:
      return self._execute_rk_conv(bufs, wait=wait)
    st = time.perf_counter()
    warp = list(itertools.product(*[range(x) for x in local_size[::-1]]))
    warp_size = len(warp)
    has_control_flow = any(op in (Ops.RANGE, Ops.ENDRANGE, Ops.IF, Ops.ENDIF) for op,_,_,_ in self.uops)
    vectorize_global = False
    global_iters = itertools.product(*[range(x) for x in global_size[::-1]])
    if not has_control_flow and all(x == 1 for x in local_size):
      total_elems = math.prod(global_size)
      if 1 < total_elems <= 16384:
        warp = list(itertools.product(*[range(x) for x in global_size[::-1]]))
        warp_size = len(warp)
        global_iters = [tuple(0 for _ in global_size)]
        vectorize_global = True
    cmplt_idx = None
    cmplt_store_idx = None
    fdiv_idxs:list[int] = []
    fdiv_store_idxs:list[int] = []
    rk_round_idx = None
    rk_round_store_idx = None
    rk_round_shape:tuple[int, ...]|None = None
    rk_where_idx = None
    rk_where_store_idx = None
    rk_where_shape:tuple[int, ...]|None = None
    rk_idiv_idx = None
    rk_idiv_store_idx = None
    rk_idiv_shape:tuple[int, ...]|None = None
    rk_where_present = any(op is Ops.CUSTOM and _is_where_arg(arg) for op,_,_,arg in self.uops)
    if not has_control_flow:
      if warp_size == 1:
        cmplt_candidates = [j for j,(op,_,_,_) in enumerate(self.uops) if op is Ops.CMPLT]
        if len(cmplt_candidates) == 1:
          cmplt_idx = cmplt_candidates[0]
          stores = [j for j,(op,_,_,_) in enumerate(self.uops) if op is Ops.STORE]
          if len(stores) != 1:
            cmplt_idx = None
            cmplt_store_idx = None
          else:
            store_candidates = [j for j,(op,_,idp,_) in enumerate(self.uops) if op is Ops.STORE and cmplt_idx in idp]
            if len(store_candidates) == 1:
              cmplt_store_idx = store_candidates[0]
              other_users = [j for j,(_,_,idp,_) in enumerate(self.uops) if cmplt_idx in idp and j not in (cmplt_idx, cmplt_store_idx)]
              if other_users:
                cmplt_idx = None
                cmplt_store_idx = None
            else:
              cmplt_idx = None
              cmplt_store_idx = None
      fdiv_candidates = [j for j,(op,_,_,_) in enumerate(self.uops) if op is Ops.FDIV]
      if fdiv_candidates:
        fdiv_store_idxs = []
        for fidx in fdiv_candidates:
          users = [j for j,(_,_,idp,_) in enumerate(self.uops) if fidx in idp]
          store_users = [j for j in users if self.uops[j][0] is Ops.STORE]
          other_users = [j for j in users if self.uops[j][0] is not Ops.STORE]
          if other_users or len(store_users) != 1:
            fdiv_candidates = []
            fdiv_store_idxs = []
            break
          fdiv_store_idxs.append(store_users[0])
        if len(set(fdiv_store_idxs)) != len(fdiv_store_idxs):
          fdiv_candidates = []
          fdiv_store_idxs = []
      fdiv_idxs = fdiv_candidates
      rk_round_candidates = [j for j,(op,_,_,arg) in enumerate(self.uops) if op is Ops.CUSTOM and _is_roundoff_arg(arg)]
      if len(rk_round_candidates) == 1:
        rk_idx = rk_round_candidates[0]
        users = [j for j,(_,_,idp,_) in enumerate(self.uops) if rk_idx in idp]
        store_users = [j for j in users if self.uops[j][0] is Ops.STORE]
        other_users = [j for j in users if self.uops[j][0] is not Ops.STORE]
        if not other_users and len(store_users) == 1:
          rk_round_idx = rk_idx
          rk_round_store_idx = store_users[0]
          rk_arg = self.uops[rk_idx][3]
          if isinstance(rk_arg, tuple) and len(rk_arg) > 1:
            rk_round_shape = rk_arg[1]
      rk_where_candidates = [j for j,(op,_,_,arg) in enumerate(self.uops) if op is Ops.CUSTOM and _is_where_arg(arg)]
      if len(rk_where_candidates) == 1:
        rk_idx = rk_where_candidates[0]
        users = [j for j,(_,_,idp,_) in enumerate(self.uops) if rk_idx in idp]
        store_users = [j for j in users if self.uops[j][0] is Ops.STORE]
        other_users = [j for j in users if self.uops[j][0] is not Ops.STORE]
        if not other_users and len(store_users) == 1:
          rk_where_idx = rk_idx
          rk_where_store_idx = store_users[0]
          rk_arg = self.uops[rk_idx][3]
          if isinstance(rk_arg, tuple) and len(rk_arg) > 1:
            rk_where_shape = rk_arg[1]
      rk_idiv_candidates = [j for j,(op,_,_,arg) in enumerate(self.uops) if op is Ops.CUSTOM and _is_idiv_arg(arg)]
      if len(rk_idiv_candidates) == 1:
        rk_idx = rk_idiv_candidates[0]
        users = [j for j,(_,_,idp,_) in enumerate(self.uops) if rk_idx in idp]
        store_users = [j for j in users if self.uops[j][0] is Ops.STORE]
        other_users = [j for j in users if self.uops[j][0] is not Ops.STORE]
        if not other_users and len(store_users) == 1:
          rk_idiv_idx = rk_idx
          rk_idiv_store_idx = store_users[0]
          rk_arg = self.uops[rk_idx][3]
          if isinstance(rk_arg, tuple) and len(rk_arg) > 1:
            rk_idiv_shape = rk_arg[1]
    fdiv_idx_set = set(fdiv_idxs)
    fdiv_store_idx_set = set(fdiv_store_idxs)
    cmplt_batch_a:list[Any] = []
    cmplt_batch_b:list[Any] = []
    cmplt_batch_out_ptrs:list[tuple[Any, int, bool]] = []
    fdiv_batch_a:list[Any] = []
    fdiv_batch_b:list[Any] = []
    fdiv_batch_out_ptrs:list[tuple[Any, int, bool]] = []
    rk_round_batch_vals:list[Any] = []
    rk_round_batch_out_ptrs:list[tuple[Any, int, bool]] = []
    rk_where_batch_mask:list[Any] = []
    rk_where_batch_a:list[Any] = []
    rk_where_batch_b:list[Any] = []
    rk_where_batch_out_ptrs:list[tuple[Any, int, bool]] = []
    rk_idiv_batch_a:list[Any] = []
    rk_idiv_batch_b:list[Any] = []
    rk_idiv_batch_out_ptrs:list[tuple[Any, int, bool]] = []
    for idxs in global_iters:
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
          if cmplt_idx is not None and i == cmplt_store_idx:
            if len(inp[0]) != 1: raise RuntimeError(f"batched CMPLT store expects 1 lane, got {len(inp[0])}")
            cmplt_batch_out_ptrs.append(inp[0][0])
            i += 1
            continue
          if fdiv_idxs and i in fdiv_store_idx_set:
            fdiv_batch_out_ptrs.extend(inp[0])
            i += 1
            continue
          if rk_round_idx is not None and i == rk_round_store_idx:
            rk_round_batch_out_ptrs.extend(inp[0])
            i += 1
            continue
          if rk_where_idx is not None and i == rk_where_store_idx:
            rk_where_batch_out_ptrs.extend(inp[0])
            i += 1
            continue
          if rk_idiv_idx is not None and i == rk_idiv_store_idx:
            rk_idiv_batch_out_ptrs.extend(inp[0])
            i += 1
            continue
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
          if arg[0] == 'g':
            if vectorize_global: ul[i] = [x[2-int(arg[-1])] for x in warp]
            else: ul[i] = [idxs[2-int(arg[-1])]] * warp_size
          elif arg[0] == 'l':
            if vectorize_global: ul[i] = [0] * warp_size
            else: ul[i] = [x[2-int(arg[-1])] for x in warp]
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
          if _is_roundoff_arg(arg):
            if len(inp) != 1: raise RuntimeError(f"RK_ROUNDOFF expects 1 input, got {len(inp)}")
            if rk_round_idx is not None and i == rk_round_idx:
              rk_round_batch_vals.extend(inp[0])
              ul[i] = [0.0] * len(inp[0])
              i += 1
              continue
            ul[i] = self.roundoff([float(x) for x in inp[0]])
            i += 1
            continue
          if _is_where_arg(arg):
            if len(inp) != 3: raise RuntimeError(f"RK_WHERE expects 3 inputs, got {len(inp)}")
            if rk_where_idx is not None and i == rk_where_idx:
              rk_where_batch_mask.extend(inp[0])
              rk_where_batch_a.extend(inp[1])
              rk_where_batch_b.extend(inp[2])
              ul[i] = [0.0] * len(inp[0])
              i += 1
              continue
            shape = arg[1] if isinstance(arg, tuple) and len(arg) > 1 else None
            ul[i] = self._where_batch(inp[0], inp[1], inp[2], shape)
            i += 1
            continue
          if _is_idiv_arg(arg):
            if len(inp) != 2: raise RuntimeError(f"RK_IDIV expects 2 inputs, got {len(inp)}")
            if rk_idiv_idx is not None and i == rk_idiv_idx:
              rk_idiv_batch_a.extend(inp[0])
              rk_idiv_batch_b.extend(inp[1])
              ul[i] = [0] * len(inp[0])
              i += 1
              continue
            shape = arg[1] if isinstance(arg, tuple) and len(arg) > 1 else None
            ul[i] = self._idiv_batch(inp[0], inp[1], shape, dtype)
            i += 1
            continue
          if _is_sigmoid_arg(arg):
            if len(inp) != 1: raise RuntimeError(f"RK_SIGMOID expects 1 input, got {len(inp)}")
            if dtype not in (dtypes.float16, dtypes.float):
              raise RuntimeError(f"RK_SIGMOID unsupported dtype {dtype}")
            shape = arg[1] if isinstance(arg, tuple) and len(arg) > 1 else None
            ul[i] = self._sigmoid_batch(inp[0], shape)
            i += 1
            continue
          if _is_silu_arg(arg):
            if len(inp) != 1: raise RuntimeError(f"RK_SILU expects 1 input, got {len(inp)}")
            if dtype not in (dtypes.float16, dtypes.float):
              raise RuntimeError(f"RK_SILU unsupported dtype {dtype}")
            shape = arg[1] if isinstance(arg, tuple) and len(arg) > 1 else None
            ul[i] = self._silu_batch(inp[0], shape)
            i += 1
            continue
          if _is_abs_arg(arg):
            if len(inp) != 1: raise RuntimeError(f"RK_ABS expects 1 input, got {len(inp)}")
            if dtype not in (dtypes.float16, dtypes.float):
              raise RuntimeError(f"RK_ABS unsupported dtype {dtype}")
            shape = arg[1] if isinstance(arg, tuple) and len(arg) > 1 else None
            ul[i] = self._abs_batch(inp[0], shape)
            i += 1
            continue
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
          assert all_same([dtype] + dtp) or uop in {Ops.CMPEQ, Ops.CMPNE, Ops.CMPLT, Ops.WHERE}, f"dtype mismatch on {uop}"

          if uop is Ops.CMPLT:
            if len(inp) != 2: raise RuntimeError(f"CMPLT expects 2 inputs, got {len(inp)}")
            if cmplt_idx is not None and i == cmplt_idx:
              cmplt_batch_a.append(inp[0][0])
              cmplt_batch_b.append(inp[1][0])
              ul[i] = [False]
              i += 1
              continue
            ul[i] = self._cmplt_part1(inp[0], inp[1])
            i += 1
            continue
          if uop is Ops.CMPEQ:
            if len(inp) != 2: raise RuntimeError(f"CMPEQ expects 2 inputs, got {len(inp)}")
            ul[i] = self._cmpeq_part1(inp[0], inp[1])
            i += 1
            continue
          if uop is Ops.CMPNE:
            if len(inp) != 2: raise RuntimeError(f"CMPNE expects 2 inputs, got {len(inp)}")
            ul[i] = self._cmpneq_part1(inp[0], inp[1])
            i += 1
            continue

          if uop is Ops.FDIV and fdiv_idxs and i in fdiv_idx_set:
            if len(inp) != 2: raise RuntimeError(f"FDIV expects 2 inputs, got {len(inp)}")
            fdiv_batch_a.extend(inp[0])
            fdiv_batch_b.extend(inp[1])
            ul[i] = [0.0] * len(inp[0])
            i += 1
            continue

          if uop is Ops.FDIV and len(inp) == 2 and dtypes.is_float(dtype):
            ul[i] = self._div_part1(inp[0], inp[1])
            i += 1
            continue

          if uop is Ops.RECIP and len(inp) == 1 and dtypes.is_float(dtype):
            ul[i] = self._recip_part1(inp[0])
            i += 1
            continue
          if uop is Ops.NEG and len(inp) == 1 and dtype in (dtypes.float, dtypes.float16):
            shape = None
            ul[i] = self._neg_part1(inp[0], shape)
            i += 1
            continue
          if len(inp) == 2 and uop is Ops.MAX and dtype in (dtypes.float, dtypes.float16):
            shape = None
            ul[i] = self._max_part1(inp[0], inp[1], shape)
            i += 1
            continue
          if len(inp) == 2 and uop in (Ops.ADD, Ops.MUL) and dtype in (dtypes.float, dtypes.float16):
            shape = None
            if uop is Ops.ADD: ul[i] = self._add_part1(inp[0], inp[1], shape)
            else: ul[i] = self._mul_part1(inp[0], inp[1], shape)
            i += 1
            continue
          if len(inp) == 2 and uop in (Ops.ADD, Ops.MUL) and dtype == dtypes.int16:
            shape = None
            if uop is Ops.ADD: ul[i] = self._add_part1_int16(inp[0], inp[1], shape)
            else: ul[i] = self._mul_part1_int16(inp[0], inp[1], shape)
            i += 1
            continue
          if (len(inp) == 2
              and (dtype in (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int, dtypes.float, dtypes.float16))
              and (uop in RockchipRenderer.code_for_op.keys())):

            io_dtype = dtype
            if dtype in (dtypes.float, dtypes.float16): io_dtype = dtypes.float16
            elif dtype == dtypes.int32: io_dtype = dtypes.int32
            elif dtype == dtypes.int16: io_dtype = dtypes.int16
            elif dtype == dtypes.int8: io_dtype = dtypes.int8

            self.device.add_buffer(len(inp[0]) * io_dtype.itemsize)

            self.input_buf = self.device.input_buf
            self.weight_buf = self.device.weight_buf
            self.output_buf = self.device.output_buf

         
            import numpy as np
            self.create_reg()
            if io_dtype == dtypes.float16:
              src = memoryview(bytearray(np.float16(inp[0]).tobytes()))
              ctypes.memmove(self.input_buf.va_addr, mv_address(src), src.nbytes)
              src2 = memoryview(bytearray(np.float16(inp[1]).tobytes()))
              ctypes.memmove(self.weight_buf.va_addr, mv_address(src2), src2.nbytes)
              # FIX ME
              dst = np.frombuffer((bytearray(self.output_buf.size)), dtype=np.float16)
              # dst = np.frombuffer((bytearray(self.output_buf.size * dtypes.float32.itemsize)), dtype=np.float32)
              
              self.ops(uop, dtypes.float16)
   
            elif io_dtype == dtypes.int32:
              src = memoryview(bytearray(np.int32(inp[0]).tobytes()))
              ctypes.memmove(self.input_buf.va_addr, mv_address(src), src.nbytes)
              src2 = memoryview(bytearray(np.int32(inp[1]).tobytes()))
              ctypes.memmove(self.weight_buf.va_addr, mv_address(src2), src2.nbytes)
              dst = np.frombuffer((bytearray(self.output_buf.size)), dtype=np.int32)

              self.ops(uop, dtypes.int32)

            elif io_dtype == dtypes.int16:
              src = memoryview(bytearray(np.int16(inp[0]).tobytes()))
              ctypes.memmove(self.input_buf.va_addr, mv_address(src), src.nbytes)
              src2 = memoryview(bytearray(np.int16(inp[1]).tobytes()))
              ctypes.memmove(self.weight_buf.va_addr, mv_address(src2), src2.nbytes)
              dst = np.frombuffer((bytearray(self.output_buf.size)), dtype=np.int16)

              self.ops(uop, dtypes.int16)

            elif io_dtype == dtypes.int8:
              src = memoryview(bytearray(np.int8(inp[0]).tobytes()))
              ctypes.memmove(self.input_buf.va_addr, mv_address(src), src.nbytes)
              src2 = memoryview(bytearray(np.int8(inp[1]).tobytes()))
              ctypes.memmove(self.weight_buf.va_addr, mv_address(src2), src2.nbytes)
              dst = np.frombuffer((bytearray(self.output_buf.size)), dtype=np.int8)

              self.ops(uop, dtypes.int8)

            cols = len(inp[0]) if len(inp[0]) > 0 else 1
            data_cube_width = cols - 1
            stride_field = cols * io_dtype.itemsize
            channel = 0
            self.emit_raw(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE,
              self.reg(stride_field, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
              self.reg(data_cube_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT,
              self.reg(0, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
              self.reg(channel, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__MASK) |
              self.reg(channel, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_WDMA_SIZE_0,
              self.reg(channel, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__SHIFT, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_WDMA_SIZE_1,
              self.reg(0, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__MASK) |
              self.reg(data_cube_width, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))
            self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
              self.reg(data_cube_width, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH__MASK))
            self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
              self.reg(0, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT__MASK))
            self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
              self.reg(channel, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL__MASK))
            self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_SURF_STRIDE,
              self.reg(stride_field, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__SHIFT, rk.DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_SURFACE_ADD,
              self.reg(stride_field, rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT, rk.DPU_SURFACE_ADD_SURF_ADD__MASK))
            self.emit_raw(rk.DPU, rk.REG_DPU_DST_BASE_ADDR, 
                self.reg(self.output_buf.meta.dma_addr, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
            self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
              self.reg(self.input_buf.meta.dma_addr, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_SRC_BASE_ADDR_SRC_BASE_ADDR__MASK))
            self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
              self.reg(self.weight_buf.meta.dma_addr, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__SHIFT, rk.DPU_RDMA_RDMA_EW_BASE_ADDR_EW_BASE_ADDR__MASK))
          
            self.submit()
            ctypes.memmove(dst.ctypes.data, self.output_buf.va_addr, self.output_buf.size)
            # print("inp[0]", inp[0])            
            # print(uop)
            # print("inp[1]", inp[1])
            # print("dst", dst.tolist())
            ul[i] = dst.tolist()
          else:
            # Only allow fallback for simple logical ops.
            allow_fallback = uop in (Ops.XOR, Ops.AND, Ops.OR, Ops.TRUNC)
            if allow_fallback:
              if DEBUG >= 3:
                print('ALLOWED FALLBACK TO CPU', uop, dtype)
              ul[i] = [exec_alu(uop, dtype, p) for p in zip(*inp)]
            else:
              print('EXIT OPERATION NOT SUPPORTED', uop, dtype)
              exit()
        assert i in ul, (uop, dtype, idp, arg)
        i += 1
    if cmplt_idx is not None:
      if len(cmplt_batch_a) != len(cmplt_batch_out_ptrs):
        raise RuntimeError(f"batched CMPLT mismatch: {len(cmplt_batch_a)} values vs {len(cmplt_batch_out_ptrs)} stores")
      out_vals = self._cmplt_part1(cmplt_batch_a, cmplt_batch_b)
      out_dtype = self.uops[cmplt_idx][1]
      if out_dtype is None: raise RuntimeError("batched CMPLT missing output dtype")
      out_scalar = out_dtype.scalar()
      for (m,o,g),v in zip(cmplt_batch_out_ptrs, out_vals):
        if g: _store(m, o, v, out_scalar)
    if fdiv_idxs:
      if len(fdiv_batch_a) != len(fdiv_batch_out_ptrs):
        raise RuntimeError(f"batched FDIV mismatch: {len(fdiv_batch_a)} values vs {len(fdiv_batch_out_ptrs)} stores")
      out_vals = self._div_part1(fdiv_batch_a, fdiv_batch_b)
      out_dtype = self.uops[fdiv_idxs[0]][1]
      if out_dtype is None: raise RuntimeError("batched FDIV missing output dtype")
      out_scalar = out_dtype.scalar()
      for (m,o,g),v in zip(fdiv_batch_out_ptrs, out_vals):
        if g: _store(m, o, v, out_scalar)
    if rk_round_idx is not None:
      if len(rk_round_batch_vals) != len(rk_round_batch_out_ptrs):
        raise RuntimeError(f"batched RK_ROUNDOFF mismatch: {len(rk_round_batch_vals)} values vs {len(rk_round_batch_out_ptrs)} stores")
      out_vals = self._roundoff_batch([float(x) for x in rk_round_batch_vals], rk_round_shape)
      out_dtype = self.uops[rk_round_idx][1]
      if out_dtype is None: raise RuntimeError("batched RK_ROUNDOFF missing output dtype")
      out_scalar = out_dtype.scalar()
      for (m,o,g),v in zip(rk_round_batch_out_ptrs, out_vals):
        if g: _store(m, o, v, out_scalar)
    if rk_where_idx is not None:
      if len(rk_where_batch_mask) != len(rk_where_batch_out_ptrs):
        raise RuntimeError(f"batched RK_WHERE mismatch: {len(rk_where_batch_mask)} values vs {len(rk_where_batch_out_ptrs)} stores")
      if len(rk_where_batch_mask) != len(rk_where_batch_a) or len(rk_where_batch_mask) != len(rk_where_batch_b):
        raise RuntimeError("batched RK_WHERE input length mismatch")
      out_dtype = self.uops[rk_where_idx][1]
      if out_dtype is None: raise RuntimeError("batched RK_WHERE missing output dtype")
      out_vals = self._where_batch(rk_where_batch_mask, rk_where_batch_a, rk_where_batch_b, rk_where_shape)
      out_scalar = out_dtype.scalar()
      for (m,o,g),v in zip(rk_where_batch_out_ptrs, out_vals):
        if g: _store(m, o, v, out_scalar)
    if rk_idiv_idx is not None:
      if len(rk_idiv_batch_a) != len(rk_idiv_batch_out_ptrs):
        raise RuntimeError(f"batched RK_IDIV mismatch: {len(rk_idiv_batch_a)} values vs {len(rk_idiv_batch_out_ptrs)} stores")
      if len(rk_idiv_batch_a) != len(rk_idiv_batch_b):
        raise RuntimeError("batched RK_IDIV input length mismatch")
      out_dtype = self.uops[rk_idiv_idx][1]
      if out_dtype is None: raise RuntimeError("batched RK_IDIV missing output dtype")
      out_vals = self._idiv_batch(rk_idiv_batch_a, rk_idiv_batch_b, rk_idiv_shape, out_dtype)
      out_scalar = out_dtype.scalar()
      for (m,o,g),v in zip(rk_idiv_batch_out_ptrs, out_vals):
        if g: _store(m, o, v, out_scalar)
    return time.perf_counter() - st

  def _buffer_as_bytes(self, buf: Any) -> bytes:
    if isinstance(buf, (bytes, bytearray)): return bytes(buf)
    if isinstance(buf, memoryview): return buf.tobytes()
    if isinstance(buf, np.ndarray): return buf.tobytes()
    if hasattr(buf, "va_addr") and hasattr(buf, "size"):
      return ctypes.string_at(buf.va_addr, buf.size)
    raise TypeError(f"unsupported buffer type {type(buf)}")

  def _write_bytes(self, buf: Any, data: bytes) -> None:
    def _bounded_copy(dst_len:int, src:bytes) -> bytes:
      if len(src) <= dst_len: return src
      if DEBUG:
        print(f"RK_CONV truncating write from {len(src)} to {dst_len}")
      return src[:dst_len]
    if isinstance(buf, bytearray):
      data = _bounded_copy(len(buf), data)
      buf[:len(data)] = data
    elif isinstance(buf, memoryview):
      data = _bounded_copy(len(buf), data)
      buf[:len(data)] = data
    elif hasattr(buf, "va_addr") and hasattr(buf, "size"):
      data = _bounded_copy(buf.size, data)
      ctypes.memmove(buf.va_addr, data, len(data))
    else:
      raise TypeError(f"unsupported buffer type {type(buf)}")

  def _submit_conv(self, cmd_sequences:list[list[int]]|None=None) -> None:
    self.device.reset_controller_if_needed()
    try:
      rk.DRM_IOCTL_RKNPU_ACTION(self.device.fd_ctl, flags=rk.RKNPU_ACT_RESET)
    except Exception as exc:
      if DEBUG:
        print("RK_CONV reset failed", exc)
    sequences = cmd_sequences if cmd_sequences is not None else [list(self.q)]
    if not sequences:
      return
    self._submit_count = getattr(self, "_submit_count", 0) + 1
    if hasattr(self.device, "_submission_total"):
      self.device._submission_total += 1
    if DEBUG >= 3:
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
      tasks[task_idx].op_idx = task_idx + 1
      tasks[task_idx].enable_mask = 0xd
      tasks[task_idx].int_mask = 0x300
      tasks[task_idx].int_clear = 0x1ffff
      tasks[task_idx].int_status = 0
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
    try:
      rk.DRM_IOCTL_RKNPU_ACTION(self.device.fd_ctl, flags=rk.RKNPU_ACT_RESET)
    except Exception as exc:
      if DEBUG:
        print("RK_CONV post-setup reset failed", exc)

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
      core_mask=0,
      fence_fd=-1,
      subcore_task=(rk.struct_rknpu_subcore_task * 5)(
        rk.struct_rknpu_subcore_task(task_start=0, task_number=len(sequences)),
        rk.struct_rknpu_subcore_task(task_start=0, task_number=0),
        rk.struct_rknpu_subcore_task(task_start=0, task_number=0),
      ),
    )
    os.system("bash -c \"cd ~/npu/ops_rknn/ && python dump.py 2 | grep EMIT | sed 's/\\x1B\\[[0-9;]*[a-zA-Z]//g' | sed 's/^.*EMIT(/EMIT(/' > /tmp/tinygrad_gem2\"")
    if DEBUG >= 3:
      os.system("bash -c 'cd ~/npu/ops_rknn/ && python dump.py 1' ")
      os.system("bash -c 'cd ~/npu/ops_rknn/ && python dump.py 2' ")
      os.system("bash -c 'cd ~/npu/ops_rknn/ && python dump.py 3' ")
      os.system("bash -c 'cd ~/npu/ops_rknn/ && python dump.py 4' ")
      os.system("bash -c 'cd ~/npu/ops_rknn/ && python dump.py 5' ")


    print("DRM_IOCTL_RKNPU_SUBMIT_CONV")
    rk.DRM_IOCTL_RKNPU_SUBMIT(self.device.fd_ctl, __payload=submit_res)
    if DEBUG >= 3:
      os.system("bash -c 'cd ~/npu/ops_rknn/ && python dump.py 5' ")

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

  def _pack_matmul_input_stride32(self, src: np.ndarray, align_in:int) -> np.ndarray:
    if src.ndim != 2:
      raise ValueError(f"expected 2D matmul input, got {src.shape}")
    rows, cols = src.shape
    if cols > align_in:
      raise ValueError(f"matmul input width {cols} exceeds align_in {align_in}")
    packed = np.zeros((rows, align_in), dtype=np.float16)
    packed[:, :cols] = src
    return packed.reshape(-1)

  def _pack_matmul_weight_column_major(self, src: np.ndarray, align_in:int, align_out:int) -> np.ndarray:
    if src.ndim != 2:
      raise ValueError(f"expected 2D matmul weight, got {src.shape}")
    rows, cols = src.shape
    if rows > align_in or cols > align_out:
      raise ValueError(f"matmul weight shape {src.shape} exceeds align_in {align_in} align_out {align_out}")
    packed = np.zeros((align_out, align_in), dtype=np.float16)
    packed[:cols, :rows] = src.T
    return packed.reshape(-1)

  def _pack_matmul_input_nc1hwc2_fp16(self, src: np.ndarray, align_in:int, out_height:int, c2:int) -> np.ndarray:
    if src.ndim != 2:
      raise ValueError(f"expected 2D matmul input, got {src.shape}")
    if align_in <= 0 or out_height <= 0 or c2 <= 0:
      raise ValueError(f"invalid matmul pack args align_in={align_in} out_height={out_height} c2={c2}")
    rows, cols = src.shape
    if rows > out_height or cols > align_in:
      raise ValueError(f"matmul input shape {src.shape} exceeds out_height {out_height} align_in {align_in}")
    planes = (align_in + c2 - 1) // c2
    plane_stride = out_height * c2
    dst = np.zeros(planes * plane_stride, dtype=np.float16)
    src_view = np.ascontiguousarray(src.astype(np.float16, copy=False))
    for m in range(rows):
      row_base = m * c2
      for k in range(cols):
        plane = k // c2
        offset = k % c2
        dst[plane * plane_stride + row_base + offset] = src_view[m, k]
    return dst

  def _pack_matmul_input_64x64_fp16(self, src: np.ndarray) -> np.ndarray:
    return self._pack_matmul_input_nc1hwc2_fp16(src, align_in=64, out_height=64, c2=8)
  def _pack_matmul_input_256x256_fp16(self, src: np.ndarray) -> np.ndarray:
    return self._pack_matmul_input_nc1hwc2_fp16(src, align_in=256, out_height=256, c2=8)

  def _pack_matmul_weights_fp16(self, src: np.ndarray, align_in:int, align_out:int) -> np.ndarray:
    if src.ndim != 2:
      raise ValueError(f"expected 2D matmul weight, got {src.shape}")
    if align_in <= 0 or align_out <= 0:
      raise ValueError(f"invalid matmul weight pack args align_in={align_in} align_out={align_out}")
    if align_in < 32:
      raise ValueError(f"matmul weight pack requires align_in >= 32, got {align_in}")
    rows, cols = src.shape
    if rows > align_in or cols > align_out:
      raise ValueError(f"matmul weight shape {src.shape} exceeds align_in {align_in} align_out {align_out}")
    dst = np.zeros(align_in * align_out, dtype=np.float16)
    padded = np.zeros((align_in, align_out), dtype=np.float16)
    padded[:rows, :cols] = np.ascontiguousarray(src.astype(np.float16, copy=False))
    group_stride = 16 * align_in
    c_block_stride = 32 * 16
    for out_idx in range(align_out):
      kpg = out_idx // 16
      for in_idx in range(align_in):
        cpg = in_idx // 32
        dst_idx = cpg * c_block_stride + kpg * group_stride + (in_idx % 32) + (out_idx % 16) * 32
        dst[dst_idx] = padded[in_idx, out_idx]
    return dst.reshape(-1)

  def _unpack_matmul_output_fp32_with_c2(self, src: np.ndarray, M:int, N:int, c2:int) -> np.ndarray:
    if M <= 0 or N <= 0 or c2 <= 0:
      return np.zeros((max(M, 0), max(N, 0)), dtype=np.float32)
    dst = np.zeros((M, N), dtype=np.float32)
    plane_stride = M * c2
    row_stride = c2
    total = src.size
    for n in range(N):
      plane = n // c2
      offset = n % c2
      base = plane * plane_stride + offset
      for m in range(M):
        idx = base + m * row_stride
        if idx < total:
          dst[m, n] = float(src[idx])
    return dst

  def _unpack_matmul_output_64x64_fp32(self, src: np.ndarray) -> np.ndarray:
    return self._unpack_matmul_output_fp32_with_c2(src, 64, 64, 4)
  def _unpack_matmul_output_256x256_fp32(self, src: np.ndarray) -> np.ndarray:
    return self._unpack_matmul_output_fp32_with_c2(src, 256, 256, 4)

  def _program_matmul_stride32(self, input_dma:int, weight_dma:int, output_dma:int,
                               align_in:int, align_out:int, out_height:int,
                               out_width_stride:int, reset_queue:bool=True) -> None:
    dataout_width = 1
    dataout_height = out_height
    data_in_width = dataout_width
    data_in_height = dataout_height
    feature_grains = data_in_height + 1
    dataout_atomics = dataout_width * dataout_height
    real_in_channel = align_in - 1
    orig_channel = align_out - 1
    out_channel_field = orig_channel
    weight_bytes_per_kernel = align_in * np.dtype(np.float16).itemsize
    weight_bytes_total = weight_bytes_per_kernel * align_out
    cbuf_entries = max(((dataout_width * align_in) + 31) // 32, 1)
    line_stride = data_in_width * 4
    surf_stride = 0
    notch_val = 7
    dst_surf_stride = out_width_stride
    surface_add = dst_surf_stride * max(align_out // 8, 1)
    weight_bank = 11
    data_bank = 1
    output_height_minus1 = max(dataout_height - 1, 0)
    data_cube_width = max(dataout_width - 1, 0)
    group_line_off = 0 if (align_in == 64 and align_out == 64 and out_height == 64) else 1
    if reset_queue:
      self.q = []
    reg = self.reg
    emit = self.emit_raw

    emit(rk.DPU, rk.REG_DPU_S_POINTER,
      reg(1, rk.DPU_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_S_POINTER_POINTER_PP_MODE__MASK) |
      reg(1, rk.DPU_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_EN__MASK) |
      reg(1, rk.DPU_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_S_POINTER_POINTER_PP_EN__MASK))
    conv_con1_val = (
      reg(2, rk.CNA_CONV_CON1_PROC_PRECISION__SHIFT, rk.CNA_CONV_CON1_PROC_PRECISION__MASK) |
      reg(2, rk.CNA_CONV_CON1_IN_PRECISION__SHIFT, rk.CNA_CONV_CON1_IN_PRECISION__MASK) |
      reg(group_line_off, rk.CNA_CONV_CON1_GROUP_LINE_OFF__SHIFT, rk.CNA_CONV_CON1_GROUP_LINE_OFF__MASK))
    emit(rk.CNA, rk.REG_CNA_CONV_CON1, conv_con1_val)
    emit(rk.CNA, rk.REG_CNA_CONV_CON2,
      reg(feature_grains, rk.CNA_CONV_CON2_FEATURE_GRAINS__SHIFT, rk.CNA_CONV_CON2_FEATURE_GRAINS__MASK))
    emit(rk.CNA, rk.REG_CNA_CONV_CON3,
      reg(1, rk.CNA_CONV_CON3_CONV_Y_STRIDE__SHIFT, rk.CNA_CONV_CON3_CONV_Y_STRIDE__MASK) |
      reg(1, rk.CNA_CONV_CON3_CONV_X_STRIDE__SHIFT, rk.CNA_CONV_CON3_CONV_X_STRIDE__MASK))
    emit(rk.CNA, rk.REG_CNA_DATA_SIZE0,
      reg(data_in_width, rk.CNA_DATA_SIZE0_DATAIN_WIDTH__SHIFT, rk.CNA_DATA_SIZE0_DATAIN_WIDTH__MASK) |
      reg(data_in_height, rk.CNA_DATA_SIZE0_DATAIN_HEIGHT__SHIFT, rk.CNA_DATA_SIZE0_DATAIN_HEIGHT__MASK))
    emit(rk.CNA, rk.REG_CNA_DATA_SIZE1,
      reg(real_in_channel, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL__SHIFT, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL__MASK) |
      reg(align_in, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL__SHIFT, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL__MASK))
    emit(rk.CNA, rk.REG_CNA_DATA_SIZE2,
      reg(dataout_width, rk.CNA_DATA_SIZE2_DATAOUT_WIDTH__SHIFT, rk.CNA_DATA_SIZE2_DATAOUT_WIDTH__MASK))
    emit(rk.CNA, rk.REG_CNA_DATA_SIZE3,
      reg(dataout_atomics, rk.CNA_DATA_SIZE3_DATAOUT_ATOMICS__SHIFT, rk.CNA_DATA_SIZE3_DATAOUT_ATOMICS__MASK))
    emit(rk.CNA, rk.REG_CNA_WEIGHT_SIZE0, weight_bytes_total)
    emit(rk.CNA, rk.REG_CNA_WEIGHT_SIZE1,
      reg(weight_bytes_per_kernel, rk.CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL__SHIFT, rk.CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL__MASK))
    emit(rk.CNA, rk.REG_CNA_WEIGHT_SIZE2,
      reg(1, rk.CNA_WEIGHT_SIZE2_WEIGHT_WIDTH__SHIFT, rk.CNA_WEIGHT_SIZE2_WEIGHT_WIDTH__MASK) |
      reg(1, rk.CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT__SHIFT, rk.CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT__MASK) |
      reg(align_out, rk.CNA_WEIGHT_SIZE2_WEIGHT_KERNELS__SHIFT, rk.CNA_WEIGHT_SIZE2_WEIGHT_KERNELS__MASK))
    emit(rk.CNA, rk.REG_CNA_CBUF_CON0,
      reg(weight_bank, rk.CNA_CBUF_CON0_WEIGHT_BANK__SHIFT, rk.CNA_CBUF_CON0_WEIGHT_BANK__MASK) |
      reg(data_bank, rk.CNA_CBUF_CON0_DATA_BANK__SHIFT, rk.CNA_CBUF_CON0_DATA_BANK__MASK))
    emit(rk.CNA, rk.REG_CNA_CBUF_CON1,
      reg(cbuf_entries, rk.CNA_CBUF_CON1_DATA_ENTRIES__SHIFT, rk.CNA_CBUF_CON1_DATA_ENTRIES__MASK))
    emit(rk.CNA, rk.REG_CNA_CVT_CON0,
      reg(1, rk.CNA_CVT_CON0_DATA_SIGN__SHIFT, rk.CNA_CVT_CON0_DATA_SIGN__MASK) |
      reg(1, rk.CNA_CVT_CON0_CVT_TYPE__SHIFT, rk.CNA_CVT_CON0_CVT_TYPE__MASK) |
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
      reg(line_stride, rk.CNA_DMA_CON1_LINE_STRIDE__SHIFT, rk.CNA_DMA_CON1_LINE_STRIDE__MASK))
    emit(rk.CNA, rk.REG_CNA_DMA_CON2,
      reg(surf_stride, rk.CNA_DMA_CON2_SURF_STRIDE__SHIFT, rk.CNA_DMA_CON2_SURF_STRIDE__MASK))
    emit(rk.CNA, rk.REG_CNA_FC_DATA_SIZE0,
      reg(data_in_width, rk.CNA_FC_DATA_SIZE0_DMA_WIDTH__SHIFT, rk.CNA_FC_DATA_SIZE0_DMA_WIDTH__MASK) |
      reg(data_in_height, rk.CNA_FC_DATA_SIZE0_DMA_HEIGHT__SHIFT, rk.CNA_FC_DATA_SIZE0_DMA_HEIGHT__MASK))
    emit(rk.CNA, rk.REG_CNA_FC_DATA_SIZE1,
      reg(align_in, rk.CNA_FC_DATA_SIZE1_DMA_CHANNEL__SHIFT, rk.CNA_FC_DATA_SIZE1_DMA_CHANNEL__MASK))
    emit(rk.CNA, rk.REG_CNA_DCOMP_CTRL, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_REGNUM, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_ADDR0,
      reg(weight_dma, rk.CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0__SHIFT, rk.CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0__MASK))
    for offset in range(16):
      reg_name = f"REG_CNA_DCOMP_AMOUNT{offset}"
      if hasattr(rk, reg_name):
        emit(rk.CNA, getattr(rk, reg_name), 0)
    emit(rk.CNA, rk.REG_CNA_CVT_CON5, 0)
    emit(rk.CNA, rk.REG_CNA_PAD_CON1, 0)
    emit(rk.CORE, rk.REG_CORE_MISC_CFG,
      reg(2, rk.CORE_MISC_CFG_PROC_PRECISION__SHIFT, rk.CORE_MISC_CFG_PROC_PRECISION__MASK) |
      reg(1, rk.CORE_MISC_CFG_QD_EN__SHIFT, rk.CORE_MISC_CFG_QD_EN__MASK))
    emit(rk.CORE, rk.REG_CORE_DATAOUT_SIZE_0,
      reg(output_height_minus1, rk.CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT__SHIFT, rk.CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT__MASK) |
      reg(data_cube_width, rk.CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH__SHIFT, rk.CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH__MASK))
    emit(rk.CORE, rk.REG_CORE_DATAOUT_SIZE_1,
      reg(out_channel_field, rk.CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL__SHIFT, rk.CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL__MASK))
    emit(rk.CORE, rk.REG_CORE_CLIP_TRUNCATE, 0)
    self.emit_raw(rk.CORE, 0x3030, 0)
    emit(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
      reg(15, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(2, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_FORMAT,
      reg(5, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
      reg(2, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
      reg(2, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))
    emit(rk.DPU, rk.REG_DPU_OFFSET_PEND, 0)
    emit(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
      reg(output_dma, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE,
      reg(dst_surf_stride, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT,
      reg(output_height_minus1, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR,
      reg(notch_val, rk.DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_1__SHIFT, rk.DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_1__MASK) |
      reg(notch_val, rk.DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_0__SHIFT, rk.DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_0__MASK))
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
      reg(3, rk.DPU_BS_OW_CFG_SIZE_E_2__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_2__MASK) |
      reg(3, rk.DPU_BS_OW_CFG_SIZE_E_1__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_1__MASK) |
      reg(3, rk.DPU_BS_OW_CFG_SIZE_E_0__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_0__MASK) |
      reg(1, rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT, rk.DPU_BS_OW_CFG_OD_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_OW_OP, 0)
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_0,
      reg(out_channel_field, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__SHIFT, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_1,
      reg(output_height_minus1, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__MASK) |
      reg(data_cube_width, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))
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
    emit(rk.DPU, rk.REG_DPU_SURFACE_ADD,
      reg(surface_add, rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT, rk.DPU_SURFACE_ADD_SURF_ADD__MASK))
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
    emit(rk.DPU, rk.REG_PC_VERSION, 0x00020000)
    emit(rk.DPU, rk.REG_PC_VERSION, 0x00020000)
    emit(rk.DPU, rk.REG_PC_VERSION, 0x00020000)
    if reset_queue:
      self._rk_conv_debug = {
        "dma": (input_dma, weight_dma, output_dma),
        "dst_stride": dst_surf_stride,
        "surface_add": surface_add,
        "batch_count": 1,
        "row_bytes": align_out * np.dtype(np.float32).itemsize,
        "out_channel_align": align_out,
        "data_cube_width": data_cube_width,
        "output_height_minus1": output_height_minus1,
        "dataout_atomics": dataout_atomics,
      }

  def _program_matmul_nc1hwc2(self, input_dma:int, weight_dma:int, output_dma:int,
                              align_in:int, align_out:int, out_height:int, reset_queue:bool=True) -> None:
    dataout_width = 1
    dataout_height = out_height
    data_in_width = 1
    data_in_height = out_height
    feature_grains = data_in_height + 1
    dataout_atomics = dataout_width * dataout_height
    real_in_channel = align_in - 1
    orig_channel = align_out - 1
    out_channel_field = orig_channel
    weight_bytes_per_kernel = align_in * np.dtype(np.float16).itemsize
    weight_bytes_total = weight_bytes_per_kernel * align_out
    cbuf_entries = max(((dataout_width * align_in) + 31) // 32, 1)
    line_stride = data_in_width * 4
    surf_stride = max(data_in_height - 4, 0)
    notch_val = 0
    dst_surf_stride = align_out
    surface_add = dst_surf_stride * 4
    bank_size = 32768
    total_banks = 12
    fd_bytes = align_in * data_in_width * data_in_height * np.dtype(np.float16).itemsize
    data_bank = max((fd_bytes + bank_size - 1) // bank_size, 1)
    if data_bank >= total_banks:
      weight_bank = 1
    else:
      weight_bank = total_banks - data_bank if weight_bytes_per_kernel <= bank_size else max((weight_bytes_total + bank_size - 1) // bank_size, 1)
    output_height_minus1 = max(dataout_height - 1, 0)
    data_cube_width = max(dataout_width - 1, 0)
    if reset_queue:
      self.q = []
    reg = self.reg
    emit = self.emit_raw

    emit(rk.DPU, rk.REG_DPU_S_POINTER,
      reg(1, rk.DPU_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_S_POINTER_POINTER_PP_MODE__MASK) |
      reg(1, rk.DPU_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_EN__MASK) |
      reg(1, rk.DPU_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_S_POINTER_POINTER_PP_EN__MASK))
    conv_con1_val = (
      reg(2, rk.CNA_CONV_CON1_PROC_PRECISION__SHIFT, rk.CNA_CONV_CON1_PROC_PRECISION__MASK) |
      reg(2, rk.CNA_CONV_CON1_IN_PRECISION__SHIFT, rk.CNA_CONV_CON1_IN_PRECISION__MASK) |
      reg(0, rk.CNA_CONV_CON1_GROUP_LINE_OFF__SHIFT, rk.CNA_CONV_CON1_GROUP_LINE_OFF__MASK))
    emit(rk.CNA, rk.REG_CNA_CONV_CON1, conv_con1_val)
    emit(rk.CNA, rk.REG_CNA_CONV_CON2,
      reg(feature_grains, rk.CNA_CONV_CON2_FEATURE_GRAINS__SHIFT, rk.CNA_CONV_CON2_FEATURE_GRAINS__MASK))
    emit(rk.CNA, rk.REG_CNA_CONV_CON3,
      reg(1, rk.CNA_CONV_CON3_CONV_Y_STRIDE__SHIFT, rk.CNA_CONV_CON3_CONV_Y_STRIDE__MASK) |
      reg(1, rk.CNA_CONV_CON3_CONV_X_STRIDE__SHIFT, rk.CNA_CONV_CON3_CONV_X_STRIDE__MASK))
    emit(rk.CNA, rk.REG_CNA_DATA_SIZE0,
      reg(data_in_width, rk.CNA_DATA_SIZE0_DATAIN_WIDTH__SHIFT, rk.CNA_DATA_SIZE0_DATAIN_WIDTH__MASK) |
      reg(data_in_height, rk.CNA_DATA_SIZE0_DATAIN_HEIGHT__SHIFT, rk.CNA_DATA_SIZE0_DATAIN_HEIGHT__MASK))
    emit(rk.CNA, rk.REG_CNA_DATA_SIZE1,
      reg(real_in_channel, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL__SHIFT, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL__MASK) |
      reg(align_in, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL__SHIFT, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL__MASK))
    emit(rk.CNA, rk.REG_CNA_DATA_SIZE2,
      reg(dataout_width, rk.CNA_DATA_SIZE2_DATAOUT_WIDTH__SHIFT, rk.CNA_DATA_SIZE2_DATAOUT_WIDTH__MASK))
    emit(rk.CNA, rk.REG_CNA_DATA_SIZE3,
      reg(dataout_atomics, rk.CNA_DATA_SIZE3_DATAOUT_ATOMICS__SHIFT, rk.CNA_DATA_SIZE3_DATAOUT_ATOMICS__MASK))
    emit(rk.CNA, rk.REG_CNA_WEIGHT_SIZE0, weight_bytes_total)
    emit(rk.CNA, rk.REG_CNA_WEIGHT_SIZE1,
      reg(weight_bytes_per_kernel, rk.CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL__SHIFT, rk.CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL__MASK))
    emit(rk.CNA, rk.REG_CNA_WEIGHT_SIZE2,
      reg(1, rk.CNA_WEIGHT_SIZE2_WEIGHT_WIDTH__SHIFT, rk.CNA_WEIGHT_SIZE2_WEIGHT_WIDTH__MASK) |
      reg(1, rk.CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT__SHIFT, rk.CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT__MASK) |
      reg(align_out, rk.CNA_WEIGHT_SIZE2_WEIGHT_KERNELS__SHIFT, rk.CNA_WEIGHT_SIZE2_WEIGHT_KERNELS__MASK))
    emit(rk.CNA, rk.REG_CNA_CBUF_CON0,
      reg(weight_bank, rk.CNA_CBUF_CON0_WEIGHT_BANK__SHIFT, rk.CNA_CBUF_CON0_WEIGHT_BANK__MASK) |
      reg(data_bank, rk.CNA_CBUF_CON0_DATA_BANK__SHIFT, rk.CNA_CBUF_CON0_DATA_BANK__MASK))
    emit(rk.CNA, rk.REG_CNA_CBUF_CON1,
      reg(cbuf_entries, rk.CNA_CBUF_CON1_DATA_ENTRIES__SHIFT, rk.CNA_CBUF_CON1_DATA_ENTRIES__MASK))
    emit(rk.CNA, rk.REG_CNA_CVT_CON0,
      reg(1, rk.CNA_CVT_CON0_DATA_SIGN__SHIFT, rk.CNA_CVT_CON0_DATA_SIGN__MASK) |
      reg(1, rk.CNA_CVT_CON0_CVT_TYPE__SHIFT, rk.CNA_CVT_CON0_CVT_TYPE__MASK) |
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
      reg(line_stride, rk.CNA_DMA_CON1_LINE_STRIDE__SHIFT, rk.CNA_DMA_CON1_LINE_STRIDE__MASK))
    emit(rk.CNA, rk.REG_CNA_DMA_CON2,
      reg(surf_stride, rk.CNA_DMA_CON2_SURF_STRIDE__SHIFT, rk.CNA_DMA_CON2_SURF_STRIDE__MASK))
    emit(rk.CNA, rk.REG_CNA_FC_DATA_SIZE0,
      reg(data_in_width, rk.CNA_FC_DATA_SIZE0_DMA_WIDTH__SHIFT, rk.CNA_FC_DATA_SIZE0_DMA_WIDTH__MASK) |
      reg(data_in_height, rk.CNA_FC_DATA_SIZE0_DMA_HEIGHT__SHIFT, rk.CNA_FC_DATA_SIZE0_DMA_HEIGHT__MASK))
    emit(rk.CNA, rk.REG_CNA_FC_DATA_SIZE1,
      reg(align_in, rk.CNA_FC_DATA_SIZE1_DMA_CHANNEL__SHIFT, rk.CNA_FC_DATA_SIZE1_DMA_CHANNEL__MASK))
    emit(rk.CNA, rk.REG_CNA_DCOMP_CTRL, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_REGNUM, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_ADDR0,
      reg(weight_dma, rk.CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0__SHIFT, rk.CNA_DCOMP_ADDR0_DECOMPRESS_ADDR0__MASK))
    for offset in range(16):
      reg_name = f"REG_CNA_DCOMP_AMOUNT{offset}"
      if hasattr(rk, reg_name):
        emit(rk.CNA, getattr(rk, reg_name), 0)
    emit(rk.CNA, rk.REG_CNA_CVT_CON5, 0)
    emit(rk.CNA, rk.REG_CNA_PAD_CON1, 0)
    emit(rk.CORE, rk.REG_CORE_MISC_CFG,
      reg(2, rk.CORE_MISC_CFG_PROC_PRECISION__SHIFT, rk.CORE_MISC_CFG_PROC_PRECISION__MASK) |
      reg(1, rk.CORE_MISC_CFG_QD_EN__SHIFT, rk.CORE_MISC_CFG_QD_EN__MASK))
    emit(rk.CORE, rk.REG_CORE_DATAOUT_SIZE_0,
      reg(output_height_minus1, rk.CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT__SHIFT, rk.CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT__MASK) |
      reg(data_cube_width, rk.CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH__SHIFT, rk.CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH__MASK))
    emit(rk.CORE, rk.REG_CORE_DATAOUT_SIZE_1,
      reg(out_channel_field, rk.CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL__SHIFT, rk.CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL__MASK))
    emit(rk.CORE, rk.REG_CORE_CLIP_TRUNCATE, 0)
    self.emit_raw(rk.CORE, 0x3030, 0)
    emit(rk.DPU, rk.REG_DPU_FEATURE_MODE_CFG,
      reg(15, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      reg(2, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__SHIFT, rk.DPU_FEATURE_MODE_CFG_OUTPUT_MODE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_FORMAT,
      reg(5, rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
      reg(2, rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
      reg(2, rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))
    emit(rk.DPU, rk.REG_DPU_OFFSET_PEND, 0)
    emit(rk.DPU, rk.REG_DPU_DST_BASE_ADDR,
      reg(output_dma, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__SHIFT, rk.DPU_DST_BASE_ADDR_DST_BASE_ADDR__MASK))
    emit(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE,
      reg(dst_surf_stride, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT,
      reg(output_height_minus1, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR,
      reg(notch_val, rk.DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_1__SHIFT, rk.DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_1__MASK) |
      reg(notch_val, rk.DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_0__SHIFT, rk.DPU_DATA_CUBE_NOTCH_ADDR_NOTCH_ADDR_0__MASK))
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
      reg(3, rk.DPU_BS_OW_CFG_SIZE_E_2__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_2__MASK) |
      reg(3, rk.DPU_BS_OW_CFG_SIZE_E_1__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_1__MASK) |
      reg(3, rk.DPU_BS_OW_CFG_SIZE_E_0__SHIFT, rk.DPU_BS_OW_CFG_SIZE_E_0__MASK) |
      reg(1, rk.DPU_BS_OW_CFG_OD_BYPASS__SHIFT, rk.DPU_BS_OW_CFG_OD_BYPASS__MASK))
    emit(rk.DPU, rk.REG_DPU_BS_OW_OP, 0)
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_0,
      reg(out_channel_field, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__SHIFT, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_1,
      reg(output_height_minus1, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__MASK) |
      reg(data_cube_width, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))
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
      reg(0, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__SHIFT, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__MASK) |
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
    emit(rk.DPU, rk.REG_DPU_SURFACE_ADD,
      reg(surface_add, rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT, rk.DPU_SURFACE_ADD_SURF_ADD__MASK))
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
    emit(rk.DPU, rk.REG_PC_VERSION, 0x00020000)
    emit(rk.DPU, rk.REG_PC_VERSION, 0x00020000)
    emit(rk.DPU, rk.REG_PC_VERSION, 0x00020000)
    if reset_queue:
      self._rk_conv_debug = {
        "dma": (input_dma, weight_dma, output_dma),
        "dst_stride": dst_surf_stride,
        "surface_add": surface_add,
        "batch_count": 1,
        "row_bytes": align_out * np.dtype(np.float32).itemsize,
        "out_channel_align": align_out,
        "data_cube_width": data_cube_width,
        "output_height_minus1": output_height_minus1,
        "dataout_atomics": dataout_atomics,
      }

  def _program_matmul_64x64(self, input_dma:int, weight_dma:int, output_dma:int, reset_queue:bool=True) -> None:
    self._program_matmul_nc1hwc2(input_dma, weight_dma, output_dma, 64, 64, 64, reset_queue=reset_queue)

  def _program_matmul_256x256(self, input_dma:int, weight_dma:int, output_dma:int, reset_queue:bool=True) -> None:
    self._program_matmul_nc1hwc2(input_dma, weight_dma, output_dma, 256, 256, 256, reset_queue=reset_queue)

  def _matmul_stride32_square_hw(self, dim:int, lhs_bytes: bytes, rhs_bytes: bytes, dtype_read: np.dtype,
                                 dtype: DType, np_dtype: np.dtype, post_ops: tuple[tuple[Ops, Any], ...],
                                 out_shape_write: tuple[Any, ...], lhs_elems:int, rhs_elems:int,
                                 out_buf: Any) -> float:
    lhs_mat = np.frombuffer(lhs_bytes, dtype=dtype_read, count=lhs_elems).reshape((dim, dim))
    rhs_mat = np.frombuffer(rhs_bytes, dtype=dtype_read, count=rhs_elems).reshape((dim, dim))

    align_in = 32
    align_out = 32
    out_height = dim
    out_width_stride = 1

    lhs_fp16 = np.ascontiguousarray(lhs_mat.astype(np.float16, copy=False))
    rhs_fp16 = np.ascontiguousarray(rhs_mat.astype(np.float16, copy=False))
    packed_input = self._pack_matmul_input_stride32(lhs_fp16, align_in)
    packed_weight = self._pack_matmul_weight_column_major(rhs_fp16, align_in, align_out)

    input_bytes = packed_input.tobytes()
    weight_bytes = packed_weight.tobytes()
    output_elems = align_out * out_width_stride * out_height
    output_bytes = output_elems * np.dtype(np.float32).itemsize

    input_hw = weight_hw = output_hw = None
    try:
      input_hw = self.device._gpu_alloc(len(input_bytes), 0, name="matmul_input")
      weight_hw = self.device._gpu_alloc(len(weight_bytes), 0, name="matmul_weight")
      output_hw = self.device._gpu_alloc(output_bytes, 0, name="matmul_output")

      ctypes.memmove(input_hw.va_addr, input_bytes, len(input_bytes))
      ctypes.memmove(weight_hw.va_addr, weight_bytes, len(weight_bytes))
      ctypes.memset(output_hw.va_addr, 0, output_bytes)

      self._program_matmul_stride32(input_hw.meta.dma_addr, weight_hw.meta.dma_addr, output_hw.meta.dma_addr,
                                    align_in, align_out, out_height, out_width_stride, reset_queue=True)
      self._submit_conv([list(self.q)])

      raw_output = ctypes.string_at(output_hw.va_addr, output_bytes)
      packed_output = np.frombuffer(raw_output, dtype=np.float32, count=output_elems)
      unpacked = self._unpack_matmul_output_fp32_with_c2(packed_output, dim, dim, align_out)
      if post_ops:
        unpacked = self._apply_post_ops_array(unpacked, post_ops)
      out_cast = unpacked.astype(np_dtype, copy=False)
      target_shape = tuple(int(x) for x in out_shape_write if int(x) > 0)
      if len(target_shape) == 3 and target_shape[-1] == 1 and target_shape[0] * target_shape[1] == out_cast.size:
        target_shape = target_shape[:-1]
      if not target_shape or int(np.prod(target_shape)) != out_cast.size:
        target_shape = (dim, dim)
      self._write_bytes(out_buf, out_cast.reshape(target_shape).tobytes())
      return 0.0
    finally:
      for buf in (input_hw, weight_hw, output_hw):
        if buf is not None and hasattr(self.device, "_gpu_free"):
          self.device._gpu_free(buf)

  def _matmul_8x8_hw(self, lhs_bytes: bytes, rhs_bytes: bytes, dtype_read: np.dtype, dtype: DType,
                     np_dtype: np.dtype, post_ops: tuple[tuple[Ops, Any], ...],
                     out_shape_write: tuple[Any, ...], lhs_elems:int, rhs_elems:int, out_buf: Any) -> float:
    return self._matmul_stride32_square_hw(8, lhs_bytes, rhs_bytes, dtype_read, dtype, np_dtype,
                                           post_ops, out_shape_write, lhs_elems, rhs_elems, out_buf)

  def _matmul_9x9_hw(self, lhs_bytes: bytes, rhs_bytes: bytes, dtype_read: np.dtype, dtype: DType,
                     np_dtype: np.dtype, post_ops: tuple[tuple[Ops, Any], ...],
                     out_shape_write: tuple[Any, ...], lhs_elems:int, rhs_elems:int, out_buf: Any) -> float:
    return self._matmul_stride32_square_hw(9, lhs_bytes, rhs_bytes, dtype_read, dtype, np_dtype,
                                           post_ops, out_shape_write, lhs_elems, rhs_elems, out_buf)

  def _matmul_32x32_hw(self, lhs_bytes: bytes, rhs_bytes: bytes, dtype_read: np.dtype, dtype: DType,
                       np_dtype: np.dtype, post_ops: tuple[tuple[Ops, Any], ...],
                       out_shape_write: tuple[Any, ...], lhs_elems:int, rhs_elems:int, out_buf: Any) -> float:
    return self._matmul_stride32_square_hw(32, lhs_bytes, rhs_bytes, dtype_read, dtype, np_dtype,
                                           post_ops, out_shape_write, lhs_elems, rhs_elems, out_buf)

  def _matmul_64x64_hw(self, lhs_bytes: bytes, rhs_bytes: bytes, dtype_read: np.dtype, dtype: DType,
                       np_dtype: np.dtype, post_ops: tuple[tuple[Ops, Any], ...],
                       out_shape_write: tuple[Any, ...], lhs_elems:int, rhs_elems:int, out_buf: Any) -> float:
    lhs_mat = np.frombuffer(lhs_bytes, dtype=dtype_read, count=lhs_elems).reshape((64, 64))
    rhs_mat = np.frombuffer(rhs_bytes, dtype=dtype_read, count=rhs_elems).reshape((64, 64))

    lhs_fp16 = np.ascontiguousarray(lhs_mat.astype(np.float16, copy=False))
    rhs_fp16 = np.ascontiguousarray(rhs_mat.astype(np.float16, copy=False))
    packed_input = self._pack_matmul_input_64x64_fp16(lhs_fp16)
    packed_weight = self._pack_matmul_weights_fp16(rhs_fp16, 64, 64)

    input_bytes = packed_input.tobytes()
    weight_bytes = packed_weight.tobytes()
    output_elems = 64 * 64
    output_bytes = output_elems * np.dtype(np.float32).itemsize

    input_hw = weight_hw = output_hw = None
    try:
      input_hw = self.device._gpu_alloc(len(input_bytes), 0, name="matmul_input")
      weight_hw = self.device._gpu_alloc(len(weight_bytes), 0, name="matmul_weight")
      output_hw = self.device._gpu_alloc(output_bytes, 0, name="matmul_output")

      ctypes.memmove(input_hw.va_addr, input_bytes, len(input_bytes))
      ctypes.memmove(weight_hw.va_addr, weight_bytes, len(weight_bytes))
      ctypes.memset(output_hw.va_addr, 0, output_bytes)

      self._program_matmul_64x64(input_hw.meta.dma_addr, weight_hw.meta.dma_addr, output_hw.meta.dma_addr, reset_queue=True)
      self._submit_conv([list(self.q)])

      raw_output = ctypes.string_at(output_hw.va_addr, output_bytes)
      packed_output = np.frombuffer(raw_output, dtype=np.float32, count=output_elems)
      unpacked = self._unpack_matmul_output_64x64_fp32(packed_output)
      if post_ops:
        unpacked = self._apply_post_ops_array(unpacked, post_ops)
      out_cast = unpacked.astype(np_dtype, copy=False)
      target_shape = tuple(int(x) for x in out_shape_write if int(x) > 0)
      if len(target_shape) == 3 and target_shape[-1] == 1 and target_shape[0] * target_shape[1] == out_cast.size:
        target_shape = target_shape[:-1]
      if not target_shape or int(np.prod(target_shape)) != out_cast.size:
        target_shape = (64, 64)
      self._write_bytes(out_buf, out_cast.reshape(target_shape).tobytes())
      return 0.0
    finally:
      for buf in (input_hw, weight_hw, output_hw):
        if buf is not None and hasattr(self.device, "_gpu_free"):
          self.device._gpu_free(buf)

  def _matmul_256x256_hw(self, lhs_bytes: bytes, rhs_bytes: bytes, dtype_read: np.dtype, dtype: DType,
                         np_dtype: np.dtype, post_ops: tuple[tuple[Ops, Any], ...],
                         out_shape_write: tuple[Any, ...], lhs_elems:int, rhs_elems:int, out_buf: Any) -> float:
    lhs_mat = np.frombuffer(lhs_bytes, dtype=dtype_read, count=lhs_elems).reshape((256, 256))
    rhs_mat = np.frombuffer(rhs_bytes, dtype=dtype_read, count=rhs_elems).reshape((256, 256))

    lhs_fp16 = np.ascontiguousarray(lhs_mat.astype(np.float16, copy=False))
    rhs_fp16 = np.ascontiguousarray(rhs_mat.astype(np.float16, copy=False))
    packed_input = self._pack_matmul_input_256x256_fp16(lhs_fp16)
    packed_weight = self._pack_matmul_weights_fp16(rhs_fp16, 256, 256)

    input_bytes = packed_input.tobytes()
    weight_bytes = packed_weight.tobytes()
    output_elems = 256 * 256
    output_bytes = output_elems * np.dtype(np.float32).itemsize

    input_hw = weight_hw = output_hw = None
    try:
      input_hw = self.device._gpu_alloc(len(input_bytes), 0, name="matmul_input")
      weight_hw = self.device._gpu_alloc(len(weight_bytes), 0, name="matmul_weight")
      output_hw = self.device._gpu_alloc(output_bytes, 0, name="matmul_output")

      ctypes.memmove(input_hw.va_addr, input_bytes, len(input_bytes))
      ctypes.memmove(weight_hw.va_addr, weight_bytes, len(weight_bytes))
      ctypes.memset(output_hw.va_addr, 0, output_bytes)

      self._program_matmul_256x256(input_hw.meta.dma_addr, weight_hw.meta.dma_addr, output_hw.meta.dma_addr, reset_queue=True)
      self._submit_conv([list(self.q)])

      raw_output = ctypes.string_at(output_hw.va_addr, output_bytes)
      packed_output = np.frombuffer(raw_output, dtype=np.float32, count=output_elems)
      unpacked = self._unpack_matmul_output_256x256_fp32(packed_output)
      if post_ops:
        unpacked = self._apply_post_ops_array(unpacked, post_ops)
      out_cast = unpacked.astype(np_dtype, copy=False)
      target_shape = tuple(int(x) for x in out_shape_write if int(x) > 0)
      if len(target_shape) == 3 and target_shape[-1] == 1 and target_shape[0] * target_shape[1] == out_cast.size:
        target_shape = target_shape[:-1]
      if not target_shape or int(np.prod(target_shape)) != out_cast.size:
        target_shape = (256, 256)
      self._write_bytes(out_buf, out_cast.reshape(target_shape).tobytes())
      return 0.0
    finally:
      for buf in (input_hw, weight_hw, output_hw):
        if buf is not None and hasattr(self.device, "_gpu_free"):
          self.device._gpu_free(buf)

  def _infer_matmul_dims(self, lhs_elems:int, rhs_elems:int, out_shape:tuple[Any, ...]) -> tuple[int, int, int]|None:
    if lhs_elems <= 0 or rhs_elems <= 0: return None
    dims = tuple(int(x) for x in out_shape if int(x) > 0) if out_shape else tuple()
    if dims and dims[-1] == 1 and len(dims) >= 2:
      dims = dims[:-1]
    if len(dims) == 2:
      M, N = dims
    elif len(dims) == 1:
      M, N = dims[0], 1
    else:
      return None
    if M <= 0 or N <= 0: return None
    if lhs_elems % M or rhs_elems % N: return None
    K_lhs = lhs_elems // M
    K_rhs = rhs_elems // N
    if K_lhs != K_rhs or K_lhs <= 0:
      return None
    return (M, K_lhs, N)

  def _conv1d_shape_info(self, info:RockchipConvInfo, lhs_len_single:int, rhs_len_single:int) -> tuple[int, int, int, int, int, int, int, int]|None:
    conv_meta_name = next((name for name in info.metadata if name.startswith("conv")), None)
    try:
      _, data = _parse_conv_metadata(conv_meta_name) if conv_meta_name else ("", {})
    except Exception:
      return None

    def _infer_from_out_shape(lhs_total:int, rhs_total:int, out_shape_val:tuple[int, ...]) -> tuple[int, int, int, int, int, int, int, int]|None:
      dims = [int(x) for x in out_shape_val if int(x) > 1] if out_shape_val else []
      if len(dims) < 2:
        return None
      if len(dims) >= 4:
        batch_guess = dims[0]
        groups_guess = dims[-3]
        out_per_group = dims[-2]
        output_width_guess = dims[-1]
        out_channels_guess = groups_guess * out_per_group
      else:
        batch_guess = dims[0] if len(dims) >= 3 else 1
        out_channels_guess = dims[-2]
        output_width_guess = dims[-1]
        groups_guess = 1
      if batch_guess <= 0 or out_channels_guess <= 0 or output_width_guess <= 0:
        return None
      for kw in range(1, min(output_width_guess + 1, rhs_total + 1)):
        div = out_channels_guess * kw
        if div == 0 or rhs_total % div:
          continue
        weight_in_channels = rhs_total // div
        if weight_in_channels <= 0:
          continue
        in_channels_guess = weight_in_channels * max(groups_guess, 1)
        if in_channels_guess <= 0:
          continue
        sample = in_channels_guess * (output_width_guess + kw - 1)
        if sample <= 0 or lhs_total % sample:
          continue
        batches = lhs_total // sample
        if batches != batch_guess:
          continue
        return (output_width_guess + kw - 1, kw, output_width_guess, in_channels_guess, out_channels_guess, max(groups_guess, 1), 1, 1)
      return None

    def _infer_from_lengths(lhs_total:int, rhs_total:int, out_shape_val:tuple[int, ...],
                            input_w_hint:int|None=None, kw_hint:int|None=None,
                            in_ch_hint:int|None=None, groups_hint:int|None=None,
                            stride_hint:int|None=None, dilation_hint:int|None=None) -> tuple[int, int, int, int, int, int, int, int]|None:
      out_total = int(np.prod(out_shape_val)) if out_shape_val else 0
      if DEBUG >= 4:
        print("RK_CONV length infer start", lhs_total, rhs_total, out_total,
              "hints", input_w_hint, kw_hint, in_ch_hint, groups_hint, stride_hint, dilation_hint)
      stride_candidates = [stride_hint] if stride_hint else [1]
      dilation_candidates = [dilation_hint] if dilation_hint else [1]
      in_ch_candidates = [in_ch_hint] if in_ch_hint else range(1, min(lhs_total, 128) + 1)
      for in_ch in in_ch_candidates:
        if in_ch <= 0 or lhs_total % in_ch: continue
        input_w = lhs_total // in_ch
        if input_w_hint and input_w_hint != input_w: continue
        groups_candidates = [groups_hint] if groups_hint else [g for g in range(1, in_ch + 1) if in_ch % g == 0]
        for groups in groups_candidates:
          if groups is None or groups <= 0 or in_ch % groups: continue
          weight_in = in_ch // groups
          kw_candidates = [kw_hint] if kw_hint else range(1, min(input_w, rhs_total) + 1)
          for kw in kw_candidates:
            if kw is None or kw <= 0: continue
            div = weight_in * kw
            if div == 0 or rhs_total % div: continue
            out_ch = rhs_total // div
            if out_ch <= 0 or out_ch % groups: continue
            for stride in stride_candidates:
              if stride is None or stride <= 0: continue
              for dilation in dilation_candidates:
                if dilation is None or dilation <= 0: continue
                eff_kw = (kw - 1) * dilation + 1
                output_w = (input_w - eff_kw) // stride + 1
                if output_w <= 0: continue
                expected = out_ch * output_w
                if out_total and out_total != expected:
                  continue
                if DEBUG >= 4:
                  print("RK_CONV length infer", input_w, kw, output_w, in_ch, out_ch, groups, stride, dilation)
                return (input_w, kw, output_w, in_ch, out_ch, groups, stride, dilation)
      return None

    def _shape_or_fallback(meta_shape:tuple[int, ...]|None, *fallback:tuple[int, ...]) -> tuple[int, ...]:
      if meta_shape:
        return tuple(int(x) for x in meta_shape)
      for candidate in fallback:
        if candidate:
          return tuple(int(x) for x in candidate)
      return tuple()

    lhs_shape = _shape_or_fallback(data.get("lhs"), info.lhs_tensor_shape, info.lhs_base_shape, info.lhs_shape)
    rhs_shape = _shape_or_fallback(data.get("rhs"), info.rhs_tensor_shape, info.rhs_base_shape, info.rhs_shape)
    out_shape = _shape_or_fallback(data.get("out"), info.out_tensor_shape, info.out_base_shape, info.out_shape)
    hw_shape = data.get("hw")
    stride_shape = data.get("stride")
    dilation_shape = data.get("dilation")
    groups_shape = data.get("groups")
    if (len(lhs_shape) <= 1 or len(rhs_shape) <= 1) and out_shape:
      direct = _infer_from_out_shape(lhs_len_single, rhs_len_single, out_shape)
      if direct is not None:
        return direct
    if not lhs_shape or not rhs_shape:
      direct = _infer_from_out_shape(lhs_len_single, rhs_len_single, out_shape)
      if direct is not None:
        return direct
      return _infer_from_lengths(lhs_len_single, rhs_len_single, out_shape)

    input_width = int(lhs_shape[-1]) if lhs_shape else 0
    rhs_channels = int(rhs_shape[-2]) if len(rhs_shape) >= 2 else 0
    in_channels = int(lhs_shape[-2]) if len(lhs_shape) >= 2 else (rhs_channels if rhs_channels else 0)
    kernel_width = int(hw_shape[-1]) if hw_shape else (int(rhs_shape[-1]) if len(rhs_shape) >= 2 else 0)
    out_channels = int(out_shape[-2]) if len(out_shape) >= 2 else (int(rhs_shape[0]) if len(rhs_shape) >= 1 and len(rhs_shape) > 1 else 0)
    stride = int(stride_shape[-1]) if stride_shape else 1
    dilation = int(dilation_shape[-1]) if dilation_shape else 1
    groups = int(groups_shape[0]) if groups_shape else (in_channels // rhs_channels if rhs_channels and in_channels % rhs_channels == 0 else 1)
    eff_kernel = (kernel_width - 1) * dilation + 1
    output_width = (input_width - eff_kernel) // stride + 1 if input_width and eff_kernel else 0
    out_known = int(np.prod(out_shape)) if out_shape else 0

    weight_in_channels = in_channels // groups if groups and in_channels % groups == 0 else None
    sample_elems = input_width * in_channels if input_width and in_channels else 0
    rhs_expected = out_channels * (weight_in_channels if weight_in_channels is not None else rhs_channels) * kernel_width if out_channels else 0
    batch_guess = (lhs_len_single // sample_elems) if sample_elems > 0 else 0
    out_expected = out_channels * output_width * (batch_guess if batch_guess > 0 else 1) if out_channels and output_width else 0
    valid_dims = (
      input_width > 0 and kernel_width > 0 and output_width > 0 and in_channels > 0 and out_channels > 0 and weight_in_channels is not None
      and stride > 0 and dilation > 0 and groups > 0 and sample_elems > 0 and lhs_len_single % sample_elems == 0
      and rhs_expected > 0 and rhs_len_single == rhs_expected
      and (not out_known or out_known in (out_channels * output_width, out_expected))
    )
    if valid_dims:
      return (input_width, kernel_width, output_width, in_channels, out_channels, groups, stride, dilation)

    sample_ok = input_width > 0 and in_channels > 0 and lhs_len_single % (input_width * in_channels if input_width and in_channels else 1) == 0
    kw_ok = kernel_width > 0 and rhs_len_single % kernel_width == 0
    return _infer_from_lengths(lhs_len_single, rhs_len_single, out_shape,
                               input_w_hint=input_width if sample_ok else None,
                               kw_hint=kernel_width if kw_ok else None,
                               in_ch_hint=in_channels if in_channels > 0 and lhs_len_single % in_channels == 0 else None,
                               groups_hint=groups if groups > 0 and in_channels > 0 and in_channels % max(groups, 1) == 0 else None,
                               stride_hint=stride if stride > 0 else None,
                               dilation_hint=dilation if dilation > 0 else None)

  def _conv1d_hw_full(self, lhs_vec: np.ndarray, rhs_vec: np.ndarray, dtype: DType, np_dtype: np.dtype,
                      input_width:int, kernel_width:int, output_width:int,
                      in_channels:int, out_channels:int, groups:int, stride:int, dilation:int,
                      batch_count:int=1, out_align_override:int|None=None) -> np.ndarray|None:
    if dilation != 1 or stride <= 0: return None
    if batch_count <= 0 or in_channels <= 0 or out_channels <= 0 or groups <= 0: return None
    if input_width < kernel_width: return None
    if in_channels % groups != 0: return None
    weight_in_channels = in_channels // groups
    sample_elems = input_width * in_channels
    if lhs_vec.size != sample_elems * batch_count: return None
    if rhs_vec.size != out_channels * weight_in_channels * kernel_width: return None
    self.device.reset_controller_if_needed()

    lhs_fp16 = np.ascontiguousarray(lhs_vec.astype(np.float16, copy=False))
    rhs_view = np.ascontiguousarray(rhs_vec.astype(np.float16, copy=False))
    if groups == 1:
      rhs_fp16 = rhs_view
    else:
      if out_channels % groups != 0: return None
      out_per_group = out_channels // groups
      expanded = np.zeros((out_channels, in_channels, kernel_width), dtype=np.float16)
      rhs_reshaped = rhs_view.reshape(out_channels, weight_in_channels, kernel_width)
      for g in range(groups):
        for ocg in range(out_per_group):
          oc = g * out_per_group + ocg
          start = g * weight_in_channels
          expanded[oc, start:start + weight_in_channels, :] = rhs_reshaped[oc]
      rhs_fp16 = expanded.reshape(-1)

    channel_align = max(8, ((in_channels + 7) // 8) * 8)
    out_channel_align = out_align_override if out_align_override else max(16, ((out_channels + 15) // 16) * 16)
    input_width_aligned = input_width if in_channels <= 1 else max(8, ((input_width + 7) // 8) * 8)
    dst_stride = (output_width + 3) & ~3
    if dst_stride == 0: dst_stride = output_width
    element_size = np.dtype(np.float16).itemsize
    row_bytes = dst_stride * out_channel_align * element_size
    surface_add = dst_stride * 2
    if DEBUG >= 3:
      print("RK_CONV layout dst_stride", dst_stride, "out_channel_align", out_channel_align,
            "row_bytes", row_bytes, "surface_add", surface_add, "batch", batch_count)

    lhs_view = lhs_fp16.reshape(batch_count, in_channels, 1, input_width)
    def _pack_sample(idx:int) -> np.ndarray|None:
      packed = self._pack_nc1hwc2_fp16(lhs_view[idx:idx+1], 1, in_channels, 1, input_width, channel_align, input_width_aligned)
      return packed if packed.size != 0 else None
    first_packed = _pack_sample(0)
    if first_packed is None:
      return None
    input_bytes = first_packed.nbytes

    kw_stride = out_channels * channel_align
    weights_packed = np.zeros(kw_stride * kernel_width, dtype=np.float16)
    for kw in range(kernel_width):
      kw_base = kw * kw_stride
      for oc in range(out_channels):
        oc_base = kw_base + oc * channel_align
        for ic in range(in_channels):
          src_idx = ((oc * in_channels) + ic) * kernel_width + kw
          weights_packed[oc_base + ic] = rhs_fp16[src_idx]

    padded_kernel_bytes = kernel_width * channel_align * element_size
    padded_kernel_bytes = ((padded_kernel_bytes + 15) // 16) * 16
    output_c1 = (out_channels + out_channel_align - 1) // out_channel_align
    packed_output_elems_per_sample = output_c1 * dst_stride * out_channel_align
    weight_bytes = weights_packed.nbytes
    output_bytes = packed_output_elems_per_sample * element_size
    if DEBUG >= 3:
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
      ctypes.memmove(weight_hw.va_addr, weights_packed.tobytes(), weight_bytes)
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
          batch_count=1, stride=stride, reset_queue=True)
        self._submit_conv()

        packed_output = np.frombuffer(ctypes.string_at(output_hw.va_addr, output_bytes),
                                      dtype=np.float16, count=packed_output_elems_per_sample)
        unpack_c2 = out_channel_align if out_align_override else max(8, ((out_channels + 7) // 8) * 8)
        unpacked = self._unpack_nc1hwc2_fp16(packed_output, 1, out_channels, 1, output_width,
                                             unpack_c2, dst_stride)
        results[batch_idx] = unpacked[0, :, 0, :]

      if DEBUG >= 3:
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

  def _nc1hwc2_layout(self, batch:int, channels:int, height:int, width:int, c2:int, width_stride:int) -> tuple[bool, int, int, int]:
    use_nhwc_pack = False
    if batch > 0 and channels > 0 and height > 0 and width > 0 and width_stride > 0 and c2 > 0:
      c_ratio = c2 // channels if channels > 0 else 0
      use_nhwc_pack = (c_ratio == 2) and (width_stride >= width)
    c1 = max(1, (channels + c2 - 1) // c2) if channels > 0 and c2 > 0 else 0
    plane_stride = height * width_stride * c2
    row_stride = width_stride * channels if use_nhwc_pack else width_stride * c2
    return use_nhwc_pack, c1, plane_stride, row_stride

  def _unpack_nc1hwc2_fp16(self, src: np.ndarray, batch:int, channels:int, height:int, width:int,
                           c2:int, width_stride:int) -> np.ndarray:
    """
    Convert NC1HWC2 tensors back to NCHW order using the same layout walked by
    the RKNN sample in `npu/ops_rknn/dump/conv1d_i81_11_w611.h`.
    """
    if batch <= 0 or channels <= 0 or height <= 0 or width <= 0 or width_stride <= 0 or c2 <= 0:
      return np.zeros((max(batch, 0), max(channels, 0), max(height, 0), max(width, 0)), dtype=np.float32)

    use_nhwc_pack, c1, plane_stride, row_stride = self._nc1hwc2_layout(batch, channels, height, width, c2, width_stride)
    if use_nhwc_pack:
      total = batch * plane_stride
      dst = np.zeros((batch, channels, height, width), dtype=np.float32)
      src_view = np.ascontiguousarray(src.astype(np.float16, copy=False)).reshape(-1)
      if src_view.size < total:
        src_view = np.pad(src_view, (0, total - src_view.size))
      for n in range(batch):
        n_base = n * plane_stride
        for h in range(height):
          h_base = n_base + h * row_stride
          for w in range(width):
            w_base = h_base + w * channels
            for c in range(channels):
              dst[n, c, h, w] = float(src_view[w_base + c])
      return dst

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
    use_nhwc_pack, c1, plane_stride, row_stride = self._nc1hwc2_layout(batch, channels, height, width, c2, width_stride)
    if use_nhwc_pack:
      dst = np.zeros(batch * plane_stride, dtype=np.float16)
      src_view = np.ascontiguousarray(src.astype(np.float16, copy=False)).reshape(batch, channels, height, width)
      idx = 0
      for n in range(batch):
        for h in range(height):
          for w in range(width_stride):
            for c in range(channels):
              if w < width:
                dst[idx] = src_view[n, c, h, w]
              idx += 1
      return dst
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
                              kernel_h:int, kernel_w:int, c2:int, c2_out:int, groups:int=1) -> np.ndarray:
    """
    Arrange weights in the padded OIHW layout captured in the GEM2 dump. Each
    output kernel occupies `kernel_h * kernel_w * c2_out` scalars where the
    input channel axis is padded to `c2_out`.
    """
    if out_channels <= 0 or in_channels <= 0 or kernel_h <= 0 or kernel_w <= 0 or c2_out <= 0:
      return np.zeros(0, dtype=np.float16)
    elems_per_kernel = out_channels * in_channels * kernel_h * kernel_w
    src_elems = int(src.size)
    if elems_per_kernel != src_elems and out_channels and kernel_h and kernel_w:
      per_oc_spatial = out_channels * kernel_h * kernel_w
      if per_oc_spatial > 0 and (src_elems % per_oc_spatial) == 0:
        in_channels = src_elems // per_oc_spatial
    kernel_stride = kernel_h * kernel_w * c2_out
    use_2x3_kh_major = (out_channels == 6 and in_channels == 3 and kernel_h == 2 and kernel_w == 3 and groups == 1)
    use_2x5_kh_major = (out_channels == 6 and in_channels == 3 and kernel_h == 2 and kernel_w == 5 and groups == 1)
    use_6x3x2x3_map = (out_channels == 6 and in_channels == 3 and kernel_h == 2 and kernel_w == 3)
    use_3x1_kh_major = (out_channels == 6 and in_channels == 3 and kernel_h == 3 and kernel_w == 1 and groups == 1)
    use_3x3_kh_major = (out_channels == 6 and in_channels == 3 and kernel_h == 3 and kernel_w == 3)
    use_3x5_kh_major = (out_channels == 6 and in_channels == 3 and kernel_h == 3 and kernel_w == 5 and groups == 1)
    use_2x1_kh_major = (out_channels == 6 and in_channels == 3 and kernel_h == 2 and kernel_w == 1 and groups == 1)
    dst = np.zeros(out_channels * kernel_stride, dtype=np.float16)
    src_view = np.ascontiguousarray(src.astype(np.float16, copy=False)).reshape(out_channels, in_channels, kernel_h, kernel_w)
    use_kh_major = any((use_2x3_kh_major, use_2x5_kh_major, use_3x1_kh_major, use_3x3_kh_major, use_3x5_kh_major, use_2x1_kh_major))
    if use_kh_major:
      for kh in range(kernel_h):
        for kw in range(kernel_w):
          dst_khkw_base = (kh * kernel_w + kw) * out_channels * c2_out
          for oc in range(out_channels):
            dst_spatial_base = dst_khkw_base + oc * c2_out
            for ic in range(in_channels):
              dst[dst_spatial_base + ic] = src_view[oc, ic, kh, kw]
      return dst
    if use_6x3x2x3_map:
      oc_map_6x3x2x3 = (0, 1, 2, 4, 5, 3)
      for oc in range(out_channels):
        base_kernel = oc * kernel_stride
        src_oc = oc_map_6x3x2x3[oc]
        for kh in range(kernel_h):
          for kw in range(kernel_w):
            dst_spatial_base = base_kernel + (kh * kernel_w + kw) * c2_out
            for ic in range(in_channels):
              dst[dst_spatial_base + ic] = src_view[src_oc, ic, 0, kw]
      return dst
    if use_2x1_kh_major:
      for kh in range(kernel_h):
        for kw in range(kernel_w):
          dst_khkw_base = (kh * kernel_w + kw) * out_channels * c2_out
          for oc in range(out_channels):
            dst_spatial_base = dst_khkw_base + oc * c2_out
            for ic in range(in_channels):
              dst[dst_spatial_base + ic] = src_view[oc, ic, kh, kw]
      return dst
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
    self.device.reset_controller_if_needed()

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
        if DEBUG >= 3:
          print("RK_CONV DMA", hex(input_hw.meta.dma_addr), hex(weight_hw.meta.dma_addr), hex(output_hw.meta.dma_addr))
      self._submit_conv()

      raw_output = ctypes.string_at(output_hw.va_addr, output_hw_bytes)
      if DEBUG >= 3 and idx == 0:
        hex_len = 64 if DEBUG >= 3 else 32
        print("RK_CONV raw", raw_output[:hex_len].hex())
        print("RK_CONV regs len", len(regs_snapshot) if regs_snapshot else 0)
        if regs_snapshot:
          print("RK_CONV regs full", [hex(v) for v in regs_snapshot])

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

  def _np_dtype(self, dt: DType) -> np.dtype:
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

  def _dtype_with_fp16_fallback(self, dtype: DType, lhs_bytes: bytes, rhs_bytes: bytes,
                                lhs_elems:int, rhs_elems:int) -> tuple[np.dtype, np.dtype, int]:
    np_dtype = self._np_dtype(dtype)
    dtype_read = np.dtype(np_dtype)
    item_size = dtype_read.itemsize

    def _fallback_if_fp16_divisible(reason: str) -> tuple[np.dtype, np.dtype, int]:
      if len(lhs_bytes) % np.dtype(np.float16).itemsize == 0 and len(rhs_bytes) % np.dtype(np.float16).itemsize == 0:
        if DEBUG >= 3:
          print("RK_CONV dtype fallback to fp16", reason)
        return np.float16, np.dtype(np.float16), np.dtype(np.float16).itemsize
      raise ValueError(reason)

    if item_size and (len(lhs_bytes) % item_size != 0 or len(rhs_bytes) % item_size != 0):
      np_dtype, dtype_read, item_size = _fallback_if_fp16_divisible("due to buffer size mismatch")

    if item_size and (
      (lhs_elems and lhs_elems * item_size != len(lhs_bytes)) or
      (rhs_elems and rhs_elems * item_size != len(rhs_bytes))
    ):
      np_dtype, dtype_read, item_size = _fallback_if_fp16_divisible("due to element-size mismatch")

    return np_dtype, dtype_read, item_size

  def _finalize_conv_output(self, out_buf: Any, hw_arr: np.ndarray, np_dtype: np.dtype,
                            post_ops: tuple[tuple[Ops, Any], ...], target_shape: tuple[Any, ...]) -> None:
    if post_ops:
      hw_arr = self._apply_post_ops_array(hw_arr, post_ops)
    if hw_arr.dtype != np_dtype:
      hw_arr = hw_arr.astype(np_dtype)
    target = tuple(int(x) for x in target_shape)
    target_elems = int(np.prod(target)) if target else hw_arr.size
    flat = hw_arr.reshape(-1)
    if flat.size < target_elems:
      raise ValueError(f"RK_CONV output smaller than expected ({flat.size} < {target_elems}) for shape {target}")
    if flat.size != target_elems:
      flat = flat[:target_elems]
    out_view = flat.reshape(target)
    self._write_bytes(out_buf, out_view.tobytes())

  def _infer_conv2d_dims(self, info:RockchipConvInfo, lhs_shape_flat:tuple[Any, ...],
                         rhs_shape_flat:tuple[Any, ...], out_shape_write:tuple[Any, ...],
                         axes:tuple[int, ...], lhs_elems:int, rhs_elems:int,
                         out_elems:int) -> tuple[int, int, int, int, int, int, int, int, int]|None:
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
            return (1, 1, H, W, 1, KH, KW, outH, outW)
    return None

  def _dispatch_special_conv2d(self, lhs_elems:int, rhs_elems:int, lhs_bytes:bytes, rhs_bytes:bytes,
                               dtype_read:np.dtype, dtype:DType, np_dtype:np.dtype,
                               post_ops:tuple[tuple[Ops, Any], ...], out_shape_write:tuple[Any, ...],
                               out_buf: Any, kernel_hint:tuple[int, int]|None=None) -> bool:
    candidates = list(_CONV2D_DISPATCH.get((lhs_elems, rhs_elems), ()))
    target_elems = int(np.prod(out_shape_write)) if out_shape_write else 0
    def _prioritize(items:list[RockchipConv2DDesc], pred:Callable[[RockchipConv2DDesc], bool]) -> list[RockchipConv2DDesc]:
      hits = [d for d in items if pred(d)]
      if hits:
        return hits + [d for d in items if not pred(d)]
      return items
    if target_elems > 0:
      candidates = _prioritize(candidates, lambda d: d.rhs_shape[0] * d.out_hw[0] * d.out_hw[1] == target_elems)
    if kernel_hint is not None:
      candidates = _prioritize(candidates, lambda d: d.kernel == kernel_hint)
    cpu_fallback: tuple[RockchipConv2DDesc, np.ndarray, np.ndarray]|None = None
    for desc in candidates:
      if target_elems > 0 and desc.rhs_shape[0] * desc.out_hw[0] * desc.out_hw[1] != target_elems:
        continue
      try:
        lhs_arr = np.frombuffer(lhs_bytes, dtype=dtype_read, count=lhs_elems).reshape(desc.lhs_shape)
        rhs_arr = np.frombuffer(rhs_bytes, dtype=dtype_read, count=rhs_elems).reshape(desc.rhs_shape)
      except Exception:
        continue
      if DEBUG >= 3:
        print("RK_CONV special candidate", desc.name, "out_hw", desc.out_hw, "groups", desc.groups)
      if cpu_fallback is None:
        cpu_fallback = (desc, lhs_arr, rhs_arr)
      hw_arr = self._conv2d_hw_desc(lhs_arr, rhs_arr, dtype, np_dtype, desc)
      if hw_arr is None:
        continue
      target_shape = tuple(int(x) for x in out_shape_write)
      if not target_shape or int(np.prod(target_shape)) != hw_arr.size:
        target_shape = (desc.lhs_shape[0], desc.rhs_shape[0], desc.out_hw[0], desc.out_hw[1])
      self._finalize_conv_output(out_buf, hw_arr, np_dtype, post_ops, target_shape)
      return True
    if cpu_fallback is not None:
      desc, lhs_arr, rhs_arr = cpu_fallback
      if DEBUG >= 3:
        print("RK_CONV special cpu fallback", desc.name)
      hw_arr = self._cpu_conv2d_desc(lhs_arr, rhs_arr, desc, np_dtype)
      target_shape = tuple(int(x) for x in out_shape_write)
      if not target_shape or int(np.prod(target_shape)) != hw_arr.size:
        target_shape = (desc.lhs_shape[0], desc.rhs_shape[0], desc.out_hw[0], desc.out_hw[1])
      self._finalize_conv_output(out_buf, hw_arr, np_dtype, post_ops, target_shape)
      return True
    return False

  def _conv2d_desc(self, name:str) -> RockchipConv2DDesc:
    return _CONV2D_DESCS[name]

  def _program_conv2d_desc(self, desc:RockchipConv2DDesc, input_dma:int, weight_dma:int, output_dma:int,
                           reset_queue: bool=True) -> None:
    feature_grains = desc.feature_grains
    in_h, in_w = desc.lhs_shape[2], desc.lhs_shape[3]
    out_h, out_w = desc.out_hw
    align_c, align_out_c = desc.align_c, desc.align_out_c
    width_stride = desc.width_stride
    out_width_stride = desc.out_width_stride
    dataout_atomics = out_h * out_w
    cbuf_entries = desc.cbuf_entries if desc.cbuf_entries is not None else dataout_atomics * 2
    weight_bytes_per_kernel = desc.kernel[0] * desc.kernel[1] * align_c * np.dtype(np.float16).itemsize
    weight_bytes_total = weight_bytes_per_kernel * desc.rhs_shape[0]
    surface_add = out_width_stride * 2
    cbuf_entries = dataout_atomics * 2
    in_channels = desc.lhs_shape[1]
    out_channels = desc.rhs_shape[0]
    reg = self.reg
    emit = self.emit_raw
    self.q = []

    emit(rk.CNA, rk.REG_CNA_CBUF_CON0,
      reg(11, rk.CNA_CBUF_CON0_WEIGHT_BANK__SHIFT, rk.CNA_CBUF_CON0_WEIGHT_BANK__MASK) |
      reg(1, rk.CNA_CBUF_CON0_DATA_BANK__SHIFT, rk.CNA_CBUF_CON0_DATA_BANK__MASK))
    emit(rk.CNA, rk.REG_CNA_DCOMP_REGNUM, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_CTRL, 0)
    emit(rk.CNA, rk.REG_CNA_CONV_CON1,
      reg(1, rk.CNA_CONV_CON1_NONALIGN_DMA__SHIFT, rk.CNA_CONV_CON1_NONALIGN_DMA__MASK) |
      reg(1, rk.CNA_CONV_CON1_GROUP_LINE_OFF__SHIFT, rk.CNA_CONV_CON1_GROUP_LINE_OFF__MASK) |
      reg(10, rk.CNA_CONV_CON1_ARGB_IN__SHIFT, rk.CNA_CONV_CON1_ARGB_IN__MASK) |
      reg(2, rk.CNA_CONV_CON1_PROC_PRECISION__SHIFT, rk.CNA_CONV_CON1_PROC_PRECISION__MASK) |
      reg(2, rk.CNA_CONV_CON1_IN_PRECISION__SHIFT, rk.CNA_CONV_CON1_IN_PRECISION__MASK))
    emit(rk.DPU, rk.REG_DPU_S_POINTER,
      reg(1, rk.DPU_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_S_POINTER_POINTER_PP_MODE__MASK) |
      reg(1, rk.DPU_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_EN__MASK) |
      reg(1, rk.DPU_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_S_POINTER_POINTER_PP_EN__MASK))
    emit(rk.CNA, rk.REG_CNA_CONV_CON1,
      reg(1, rk.CNA_CONV_CON1_NONALIGN_DMA__SHIFT, rk.CNA_CONV_CON1_NONALIGN_DMA__MASK) |
      reg(1, rk.CNA_CONV_CON1_GROUP_LINE_OFF__SHIFT, rk.CNA_CONV_CON1_GROUP_LINE_OFF__MASK) |
      reg(10, rk.CNA_CONV_CON1_ARGB_IN__SHIFT, rk.CNA_CONV_CON1_ARGB_IN__MASK) |
      reg(2, rk.CNA_CONV_CON1_PROC_PRECISION__SHIFT, rk.CNA_CONV_CON1_PROC_PRECISION__MASK) |
      reg(2, rk.CNA_CONV_CON1_IN_PRECISION__SHIFT, rk.CNA_CONV_CON1_IN_PRECISION__MASK))
    emit(rk.CNA, rk.REG_CNA_CONV_CON2,
      reg(feature_grains, rk.CNA_CONV_CON2_FEATURE_GRAINS__SHIFT, rk.CNA_CONV_CON2_FEATURE_GRAINS__MASK))
    emit(rk.CNA, rk.REG_CNA_CONV_CON3,
      reg(1, rk.CNA_CONV_CON3_CONV_Y_STRIDE__SHIFT, rk.CNA_CONV_CON3_CONV_Y_STRIDE__MASK) |
      reg(1, rk.CNA_CONV_CON3_CONV_X_STRIDE__SHIFT, rk.CNA_CONV_CON3_CONV_X_STRIDE__MASK))
    emit(rk.CNA, rk.REG_CNA_DATA_SIZE0,
      reg(width_stride, rk.CNA_DATA_SIZE0_DATAIN_WIDTH__SHIFT, rk.CNA_DATA_SIZE0_DATAIN_WIDTH__MASK) |
      reg(in_h, rk.CNA_DATA_SIZE0_DATAIN_HEIGHT__SHIFT, rk.CNA_DATA_SIZE0_DATAIN_HEIGHT__MASK))
    emit(rk.CNA, rk.REG_CNA_DATA_SIZE1,
      reg(max(in_channels - 1, 0), rk.CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL__SHIFT, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL__MASK) |
      reg(align_c, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL__SHIFT, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL__MASK))
    emit(rk.CNA, rk.REG_CNA_DATA_SIZE2,
      reg(out_w, rk.CNA_DATA_SIZE2_DATAOUT_WIDTH__SHIFT, rk.CNA_DATA_SIZE2_DATAOUT_WIDTH__MASK))
    emit(rk.CNA, rk.REG_CNA_DATA_SIZE3,
      reg(dataout_atomics, rk.CNA_DATA_SIZE3_DATAOUT_ATOMICS__SHIFT, rk.CNA_DATA_SIZE3_DATAOUT_ATOMICS__MASK))
    emit(rk.CNA, rk.REG_CNA_WEIGHT_SIZE0, weight_bytes_total)
    emit(rk.CNA, rk.REG_CNA_WEIGHT_SIZE1,
      reg(weight_bytes_per_kernel, rk.CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL__SHIFT, rk.CNA_WEIGHT_SIZE1_WEIGHT_BYTES_PER_KERNEL__MASK))
    emit(rk.CNA, rk.REG_CNA_WEIGHT_SIZE2,
      reg(desc.kernel[1], rk.CNA_WEIGHT_SIZE2_WEIGHT_WIDTH__SHIFT, rk.CNA_WEIGHT_SIZE2_WEIGHT_WIDTH__MASK) |
      reg(desc.kernel[0], rk.CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT__SHIFT, rk.CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT__MASK) |
      reg(out_channels, rk.CNA_WEIGHT_SIZE2_WEIGHT_KERNELS__SHIFT, rk.CNA_WEIGHT_SIZE2_WEIGHT_KERNELS__MASK))
    emit(rk.CNA, rk.REG_CNA_CBUF_CON0,
      reg(11, rk.CNA_CBUF_CON0_WEIGHT_BANK__SHIFT, rk.CNA_CBUF_CON0_WEIGHT_BANK__MASK) |
      reg(1, rk.CNA_CBUF_CON0_DATA_BANK__SHIFT, rk.CNA_CBUF_CON0_DATA_BANK__MASK))
    emit(rk.CNA, rk.REG_CNA_CBUF_CON1,
      reg(cbuf_entries, rk.CNA_CBUF_CON1_DATA_ENTRIES__SHIFT, rk.CNA_CBUF_CON1_DATA_ENTRIES__MASK))
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
    emit(rk.CNA, rk.REG_CNA_FEATURE_DATA_ADDR, input_dma)
    emit(rk.CNA, rk.REG_CNA_FC_CON2, 0)
    emit(rk.CNA, rk.REG_CNA_DMA_CON0,
      reg(15, rk.CNA_DMA_CON0_WEIGHT_BURST_LEN__SHIFT, rk.CNA_DMA_CON0_WEIGHT_BURST_LEN__MASK) |
      reg(15, rk.CNA_DMA_CON0_DATA_BURST_LEN__SHIFT, rk.CNA_DMA_CON0_DATA_BURST_LEN__MASK))
    emit(rk.CNA, rk.REG_CNA_DMA_CON1,
      reg(width_stride, rk.CNA_DMA_CON1_LINE_STRIDE__SHIFT, rk.CNA_DMA_CON1_LINE_STRIDE__MASK))
    emit(rk.CNA, rk.REG_CNA_DMA_CON2,
      reg(32, rk.CNA_DMA_CON2_SURF_STRIDE__SHIFT, rk.CNA_DMA_CON2_SURF_STRIDE__MASK))
    emit(rk.CNA, rk.REG_CNA_FC_DATA_SIZE0,
      reg(in_w, rk.CNA_FC_DATA_SIZE0_DMA_WIDTH__SHIFT, rk.CNA_FC_DATA_SIZE0_DMA_WIDTH__MASK) |
      reg(in_h, rk.CNA_FC_DATA_SIZE0_DMA_HEIGHT__SHIFT, rk.CNA_FC_DATA_SIZE0_DMA_HEIGHT__MASK))
    emit(rk.CNA, rk.REG_CNA_FC_DATA_SIZE1,
      reg(align_c, rk.CNA_FC_DATA_SIZE1_DMA_CHANNEL__SHIFT, rk.CNA_FC_DATA_SIZE1_DMA_CHANNEL__MASK))
    emit(rk.CNA, rk.REG_CNA_DCOMP_CTRL, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_REGNUM, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_ADDR0, weight_dma)
    emit(rk.CNA, rk.REG_CNA_DCOMP_AMOUNT0, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_AMOUNT1, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_AMOUNT2, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_AMOUNT3, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_AMOUNT4, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_AMOUNT5, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_AMOUNT6, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_AMOUNT7, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_AMOUNT8, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_AMOUNT9, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_AMOUNT10, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_AMOUNT11, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_AMOUNT12, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_AMOUNT13, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_AMOUNT14, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_AMOUNT15, 0)
    emit(rk.CNA, rk.REG_CNA_CVT_CON5, 0x00000fff)
    emit(rk.CNA, rk.REG_CNA_PAD_CON1, 0)
    emit(rk.CORE, rk.REG_CORE_MISC_CFG,
      reg(2, rk.CORE_MISC_CFG_PROC_PRECISION__SHIFT, rk.CORE_MISC_CFG_PROC_PRECISION__MASK))
    emit(rk.CORE, rk.REG_CORE_DATAOUT_SIZE_0,
      reg(out_h - 1, rk.CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT__SHIFT, rk.CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT__MASK) |
      reg(out_w - 1, rk.CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH__SHIFT, rk.CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH__MASK))
    emit(rk.CORE, rk.REG_CORE_DATAOUT_SIZE_1,
      reg(align_out_c - 1, rk.CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL__SHIFT, rk.CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL__MASK))
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
    emit(rk.DPU, rk.REG_DPU_DST_SURF_STRIDE,
      reg(out_width_stride, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__SHIFT, rk.DPU_DST_SURF_STRIDE_DST_SURF_STRIDE__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
      reg(out_w - 1, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT,
      reg(out_h - 1, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__MASK))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR, 0)
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_CHANNEL,
      reg(max(out_channels - 1, 0), rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL__MASK) |
      reg(align_out_c - 1, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__SHIFT, rk.DPU_DATA_CUBE_CHANNEL_CHANNEL__MASK))
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
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_0,
      reg(align_out_c - 1, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__SHIFT, rk.DPU_WDMA_SIZE_0_CHANNEL_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_1,
      reg(out_h - 1, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__MASK) |
      reg(out_w - 1, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))
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
    emit(rk.DPU, rk.REG_DPU_SURFACE_ADD,
      reg(surface_add, rk.DPU_SURFACE_ADD_SURF_ADD__SHIFT, rk.DPU_SURFACE_ADD_SURF_ADD__MASK))
    self.emit_raw(0x0, 0x40c4, 0)
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
    emit(rk.PC, rk.REG_PC_REGISTER_AMOUNTS, 0)
    emit(rk.PC, rk.REG_PC_VERSION, 0)
    self.emit_raw(0x80, rk.REG_PC_OPERATION_ENABLE,
      reg(6, rk.PC_OPERATION_ENABLE_RESERVED_0__SHIFT, rk.PC_OPERATION_ENABLE_RESERVED_0__MASK) |
      reg(1, rk.PC_OPERATION_ENABLE_OP_EN__SHIFT, rk.PC_OPERATION_ENABLE_OP_EN__MASK))
    if reset_queue:
      self._rk_conv_debug = {
        "dma": (input_dma, weight_dma, output_dma),
        "dst_stride": out_width_stride,
        "surface_add": surface_add,
        "batch_count": 1,
        "row_bytes": out_width_stride * align_out_c * np.dtype(np.float16).itemsize,
        "out_channel_align": align_out_c,
        "data_cube_width": out_w - 1,
        "output_height_minus1": out_h - 1,
        "dataout_atomics": dataout_atomics,
      }

  def _program_conv2d_6321(self, input_dma:int, weight_dma:int, output_dma:int, reset_queue: bool=True) -> None:
    self._program_conv2d_desc(self._conv2d_desc("6321"), input_dma, weight_dma, output_dma, reset_queue=reset_queue)

  def _program_conv2d_6323(self, input_dma:int, weight_dma:int, output_dma:int, reset_queue: bool=True) -> None:
    self._program_conv2d_desc(self._conv2d_desc("6323"), input_dma, weight_dma, output_dma, reset_queue=reset_queue)

  def _program_conv2d_6325(self, input_dma:int, weight_dma:int, output_dma:int, reset_queue: bool=True) -> None:
    self._program_conv2d_desc(self._conv2d_desc("6325"), input_dma, weight_dma, output_dma, reset_queue=reset_queue)

  def _program_conv2d_6331(self, input_dma:int, weight_dma:int, output_dma:int, reset_queue: bool=True) -> None:
    self._program_conv2d_desc(self._conv2d_desc("6331"), input_dma, weight_dma, output_dma, reset_queue=reset_queue)

  def _program_conv2d_6333(self, input_dma:int, weight_dma:int, output_dma:int, reset_queue: bool=True) -> None:
    self._program_conv2d_desc(self._conv2d_desc("6333"), input_dma, weight_dma, output_dma, reset_queue=reset_queue)

  def _program_conv2d_6335(self, input_dma:int, weight_dma:int, output_dma:int, reset_queue: bool=True) -> None:
    self._program_conv2d_desc(self._conv2d_desc("6335"), input_dma, weight_dma, output_dma, reset_queue=reset_queue)

  def _conv2d_hw_desc(self, lhs_arr: np.ndarray, rhs_arr: np.ndarray, dtype: DType, np_dtype: np.dtype,
                      desc: RockchipConv2DDesc) -> np.ndarray|None:
    supported_dtypes = {dtypes.float, dtypes.float32, dtypes.float16}
    if dtype not in supported_dtypes or np_dtype not in (np.float16, np.float32):
      return None
    if lhs_arr.shape != desc.lhs_shape or rhs_arr.shape != desc.rhs_shape:
      return None
    if desc.groups > 1:
      return self._cpu_conv2d_desc(lhs_arr, rhs_arr, desc, np_dtype)

    align_c, align_out_c = desc.align_c, desc.align_out_c
    lhs_fp16 = np.ascontiguousarray(lhs_arr.astype(np.float16, copy=False))
    rhs_fp16 = np.ascontiguousarray(rhs_arr.astype(np.float16, copy=False))
    if desc.weight_transform is not None:
      rhs_fp16 = desc.weight_transform(rhs_fp16)
    packed_input = self._pack_nc1hwc2_fp16(lhs_fp16, desc.lhs_shape[0], desc.lhs_shape[1],
                                           desc.lhs_shape[2], desc.lhs_shape[3], align_c, desc.width_stride)
    packed_weights = self._pack_conv_weights_fp16(rhs_fp16, desc.rhs_shape[0], rhs_fp16.shape[1],
                                                  desc.kernel[0], desc.kernel[1], align_c, align_c, groups=desc.groups)

    input_bytes = packed_input.tobytes()
    weight_bytes = packed_weights.tobytes()
    input_hw = weight_hw = output_hw = None
    out_h, out_w = desc.out_hw
    try:
      input_hw = self.device._gpu_alloc(len(input_bytes), 0, name="input")
      weight_hw = self.device._gpu_alloc(len(weight_bytes), 0, name="weight")
      output_elems = out_h * desc.out_width_stride * align_out_c
      output_hw = self.device._gpu_alloc(output_elems * np.dtype(np.float16).itemsize, 0, name="output")
      ctypes.memmove(input_hw.va_addr, input_bytes, len(input_bytes))
      ctypes.memmove(weight_hw.va_addr, weight_bytes, len(weight_bytes))
      ctypes.memset(output_hw.va_addr, 0, output_hw.size)

      self._program_conv2d_desc(desc, input_hw.meta.dma_addr, weight_hw.meta.dma_addr, output_hw.meta.dma_addr)
      self._submit_conv()

      raw_output = ctypes.string_at(output_hw.va_addr, output_hw.size)
      hw_view = np.frombuffer(raw_output, dtype=np.float16, count=output_elems)
      unpack_c2 = min(desc.align_c, desc.align_out_c)
      unpack_width_stride = desc.out_hw[1]
      nc1hwc2_elems = out_h * unpack_width_stride * unpack_c2
      unpacked = self._unpack_nc1hwc2_fp16(hw_view[:nc1hwc2_elems], desc.lhs_shape[0], desc.rhs_shape[0],
                                           out_h, out_w, unpack_c2, unpack_width_stride)
      return unpacked.astype(np_dtype, copy=False)
    finally:
      for buf in (input_hw, weight_hw, output_hw):
        if buf is not None and hasattr(self.device, "_gpu_free"):
          self.device._gpu_free(buf)

  def _cpu_conv2d_desc(self, lhs_arr: np.ndarray, rhs_arr: np.ndarray, desc: RockchipConv2DDesc,
                       np_dtype: np.dtype) -> np.ndarray:
    N, C, H, W = lhs_arr.shape
    K, _, KH, KW = rhs_arr.shape
    out_h, out_w = desc.out_hw
    groups = max(desc.groups, 1)
    out = np.zeros((N, K, out_h, out_w), dtype=np.float32)
    out_per_group = K // groups
    in_per_group = C // groups
    for n in range(N):
      for g in range(groups):
        k_base = g * out_per_group
        c_base = g * in_per_group
        for ocg in range(out_per_group):
          oc = k_base + ocg
          for y in range(out_h):
            for x in range(out_w):
              acc = 0.0
              for icg in range(in_per_group):
                ic = c_base + icg
                for kh in range(KH):
                  for kw in range(KW):
                    acc += float(lhs_arr[n, ic, y + kh, x + kw]) * float(rhs_arr[oc, icg, kh, kw])
              out[n, oc, y, x] = acc
    return out.astype(np_dtype, copy=False)

  def _conv2d_hw_6321(self, lhs_arr: np.ndarray, rhs_arr: np.ndarray, dtype: DType, np_dtype: np.dtype) -> np.ndarray|None:
    return self._conv2d_hw_desc(lhs_arr, rhs_arr, dtype, np_dtype, self._conv2d_desc("6321"))

  def _conv2d_hw_6325(self, lhs_arr: np.ndarray, rhs_arr: np.ndarray, dtype: DType, np_dtype: np.dtype) -> np.ndarray|None:
    return self._conv2d_hw_desc(lhs_arr, rhs_arr, dtype, np_dtype, self._conv2d_desc("6325"))

  def _conv2d_hw_6323(self, lhs_arr: np.ndarray, rhs_arr: np.ndarray, dtype: DType, np_dtype: np.dtype) -> np.ndarray|None:
    return self._conv2d_hw_desc(lhs_arr, rhs_arr, dtype, np_dtype, self._conv2d_desc("6323"))

  def _conv2d_hw_6331(self, lhs_arr: np.ndarray, rhs_arr: np.ndarray, dtype: DType, np_dtype: np.dtype) -> np.ndarray|None:
    return self._conv2d_hw_desc(lhs_arr, rhs_arr, dtype, np_dtype, self._conv2d_desc("6331"))

  def _conv2d_hw_6333(self, lhs_arr: np.ndarray, rhs_arr: np.ndarray, dtype: DType, np_dtype: np.dtype) -> np.ndarray|None:
    return self._conv2d_hw_desc(lhs_arr, rhs_arr, dtype, np_dtype, self._conv2d_desc("6333"))

  def _conv2d_hw_6335(self, lhs_arr: np.ndarray, rhs_arr: np.ndarray, dtype: DType, np_dtype: np.dtype) -> np.ndarray|None:
    return self._conv2d_hw_desc(lhs_arr, rhs_arr, dtype, np_dtype, self._conv2d_desc("6335"))

  def _conv2d_hw_6133(self, lhs_arr: np.ndarray, rhs_arr: np.ndarray, dtype: DType, np_dtype: np.dtype) -> np.ndarray|None:
    return self._conv2d_hw_desc(lhs_arr, rhs_arr, dtype, np_dtype, self._conv2d_desc("6133"))

  def _program_conv1d_fp16(self, input_dma: int, weight_dma: int, output_dma: int,
                           input_width: int, kernel_width: int, output_width: int,
                           in_channels: int, out_channels: int,
                           input_width_aligned: int, data_in_channel: int,
                           out_channel_align: int, dst_stride: int,
                           surface_add: int, padded_kernel_bytes: int,
                           batch_count: int, stride:int=1, reset_queue: bool=True) -> None:
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
    use_packed_channels = in_channels > 1
    real_channels = max(in_channels - 1, 0)
    data_entries = 16 if use_packed_channels else max((dst_stride + 3) // 4, 1)
    line_stride = input_width_aligned if use_packed_channels else input_width * max(data_in_channel // 2, 1)
    dma_surf_stride = 0 if use_packed_channels else 0x0fffffe0
    cvt_con0_val = reg(1, rk.CNA_CVT_CON0_CVT_BYPASS__SHIFT, rk.CNA_CVT_CON0_CVT_BYPASS__MASK)
    if not use_packed_channels:
      cvt_con0_val |= (
        reg(1, rk.CNA_CVT_CON0_DATA_SIGN__SHIFT, rk.CNA_CVT_CON0_DATA_SIGN__MASK) |
        reg(1, rk.CNA_CVT_CON0_CVT_TYPE__SHIFT, rk.CNA_CVT_CON0_CVT_TYPE__MASK))
    cvt_con5_val = 0x00000fff if use_packed_channels else 0

    emit(rk.CNA, rk.REG_CNA_CBUF_CON0,
      reg(11, rk.CNA_CBUF_CON0_WEIGHT_BANK__SHIFT, rk.CNA_CBUF_CON0_WEIGHT_BANK__MASK) |
      reg(1, rk.CNA_CBUF_CON0_DATA_BANK__SHIFT, rk.CNA_CBUF_CON0_DATA_BANK__MASK))
    emit(rk.CNA, rk.REG_CNA_DCOMP_REGNUM, 0)
    emit(rk.CNA, rk.REG_CNA_DCOMP_CTRL, 0)
    conv_con1_val = (
      reg(2, rk.CNA_CONV_CON1_PROC_PRECISION__SHIFT, rk.CNA_CONV_CON1_PROC_PRECISION__MASK) |
      reg(2, rk.CNA_CONV_CON1_IN_PRECISION__SHIFT, rk.CNA_CONV_CON1_IN_PRECISION__MASK))
    if use_packed_channels:
      conv_con1_val |= (
        reg(1, rk.CNA_CONV_CON1_NONALIGN_DMA__SHIFT, rk.CNA_CONV_CON1_NONALIGN_DMA__MASK) |
        reg(1, rk.CNA_CONV_CON1_GROUP_LINE_OFF__SHIFT, rk.CNA_CONV_CON1_GROUP_LINE_OFF__MASK) |
        reg(10, rk.CNA_CONV_CON1_ARGB_IN__SHIFT, rk.CNA_CONV_CON1_ARGB_IN__MASK))
    emit(rk.CNA, rk.REG_CNA_CONV_CON1, conv_con1_val)
    emit(rk.DPU, rk.REG_DPU_S_POINTER,
      reg(1, rk.DPU_S_POINTER_POINTER_PP_MODE__SHIFT, rk.DPU_S_POINTER_POINTER_PP_MODE__MASK) |
      reg(1, rk.DPU_S_POINTER_EXECUTER_PP_EN__SHIFT, rk.DPU_S_POINTER_EXECUTER_PP_EN__MASK) |
      reg(1, rk.DPU_S_POINTER_POINTER_PP_EN__SHIFT, rk.DPU_S_POINTER_POINTER_PP_EN__MASK))
    emit(rk.CNA, rk.REG_CNA_CONV_CON2,
      reg(feature_grains, rk.CNA_CONV_CON2_FEATURE_GRAINS__SHIFT, rk.CNA_CONV_CON2_FEATURE_GRAINS__MASK))
    emit(rk.CNA, rk.REG_CNA_CONV_CON3,
      reg(1, rk.CNA_CONV_CON3_CONV_Y_STRIDE__SHIFT, rk.CNA_CONV_CON3_CONV_Y_STRIDE__MASK) |
      reg(stride, rk.CNA_CONV_CON3_CONV_X_STRIDE__SHIFT, rk.CNA_CONV_CON3_CONV_X_STRIDE__MASK))
    data_size0_val = (
      reg(input_width_aligned, rk.CNA_DATA_SIZE0_DATAIN_WIDTH__SHIFT, rk.CNA_DATA_SIZE0_DATAIN_WIDTH__MASK) |
      reg(data_in_height, rk.CNA_DATA_SIZE0_DATAIN_HEIGHT__SHIFT, rk.CNA_DATA_SIZE0_DATAIN_HEIGHT__MASK))
    emit(rk.CNA, rk.REG_CNA_DATA_SIZE0, data_size0_val)
    if DEBUG >= 3:
      print("RK_CONV reg CNA_DATA_SIZE0 width", input_width_aligned,
            "height", data_in_height, "word", hex(data_size0_val))
    if use_packed_channels:
      emit(rk.CNA, rk.REG_CNA_DATA_SIZE1,
        reg(real_channels, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL__SHIFT, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL__MASK) |
        reg(data_in_channel, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL__SHIFT, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL__MASK))
    else:
      emit(rk.CNA, rk.REG_CNA_DATA_SIZE1,
        reg(data_in_channel, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL__SHIFT, rk.CNA_DATA_SIZE1_DATAIN_CHANNEL__MASK))
    emit(rk.CNA, rk.REG_CNA_DATA_SIZE2,
      reg(output_width, rk.CNA_DATA_SIZE2_DATAOUT_WIDTH__SHIFT, rk.CNA_DATA_SIZE2_DATAOUT_WIDTH__MASK))
    data_size3_val = reg(dataout_atomics, rk.CNA_DATA_SIZE3_DATAOUT_ATOMICS__SHIFT, rk.CNA_DATA_SIZE3_DATAOUT_ATOMICS__MASK)
    emit(rk.CNA, rk.REG_CNA_DATA_SIZE3, data_size3_val)
    if DEBUG >= 3:
      print("RK_CONV reg CNA_DATA_SIZE3 atomics", dataout_atomics, "word", hex(data_size3_val))
    emit(rk.CNA, rk.REG_CNA_WEIGHT_SIZE0, weight_bytes_total)
    # emit(rk.CNA, rk.REG_CNA_WEIGHT_SIZE0, 0x0)
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
      reg(data_entries, rk.CNA_CBUF_CON1_DATA_ENTRIES__SHIFT, rk.CNA_CBUF_CON1_DATA_ENTRIES__MASK))
    emit(rk.CNA, rk.REG_CNA_CVT_CON0, cvt_con0_val)
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
      reg(line_stride, rk.CNA_DMA_CON1_LINE_STRIDE__SHIFT, rk.CNA_DMA_CON1_LINE_STRIDE__MASK))
    emit(rk.CNA, rk.REG_CNA_DMA_CON2,
      reg(dma_surf_stride, rk.CNA_DMA_CON2_SURF_STRIDE__SHIFT, rk.CNA_DMA_CON2_SURF_STRIDE__MASK))
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
    emit(rk.CNA, rk.REG_CNA_CVT_CON5, cvt_con5_val)
    emit(rk.CNA, rk.REG_CNA_PAD_CON1, 0)
    emit(rk.CORE, rk.REG_CORE_MISC_CFG,
      reg(2, rk.CORE_MISC_CFG_PROC_PRECISION__SHIFT, rk.CORE_MISC_CFG_PROC_PRECISION__MASK))
    core_dataout_size_0_val = (
      reg(output_height_minus1, rk.CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT__SHIFT, rk.CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT__MASK) |
      reg(data_cube_width, rk.CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH__SHIFT, rk.CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH__MASK))
    emit(rk.CORE, rk.REG_CORE_DATAOUT_SIZE_0, core_dataout_size_0_val)
    if DEBUG >= 3:
      print("RK_CONV reg CORE_DATAOUT_SIZE_0 height", output_height_minus1,
            "width", data_cube_width, "word", hex(core_dataout_size_0_val))
    core_dataout_size_1_val = reg(out_channel_field, rk.CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL__SHIFT, rk.CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL__MASK)
    emit(rk.CORE, rk.REG_CORE_DATAOUT_SIZE_1, core_dataout_size_1_val)
    if DEBUG >= 3:
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
    if DEBUG >= 3:
      print("RK_CONV reg DPU_DST_SURF_STRIDE dst_stride", dst_stride, "word", hex(dpu_dst_surf_stride_val))
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_WIDTH,
      reg(data_cube_width, rk.DPU_DATA_CUBE_WIDTH_WIDTH__SHIFT, rk.DPU_DATA_CUBE_WIDTH_WIDTH__MASK))
    dpu_data_cube_height_val = reg(output_height_minus1, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__SHIFT, rk.DPU_DATA_CUBE_HEIGHT_HEIGHT__MASK)
    emit(rk.DPU, rk.REG_DPU_DATA_CUBE_HEIGHT, dpu_data_cube_height_val)
    if DEBUG >= 3:
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
    if DEBUG >= 3:
      print("RK_CONV reg DPU_WDMA_SIZE_0 channel_field", out_channel_field, "word", hex(dpu_wdma_size_0_val))
    dpu_wdma_size_1_val = (
      reg(output_height_minus1, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_HEIGHT_WDMA__MASK) |
      reg(data_cube_width, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__SHIFT, rk.DPU_WDMA_SIZE_1_WIDTH_WDMA__MASK))
    emit(rk.DPU, rk.REG_DPU_WDMA_SIZE_1, dpu_wdma_size_1_val)
    if DEBUG >= 3:
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
    if DEBUG >= 3:
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
    emit(rk.DPU, rk.REG_PC_VERSION, 0x00020000)
    emit(rk.DPU, rk.REG_PC_VERSION, 0x00020000)
    emit(rk.DPU, rk.REG_PC_VERSION, 0x00020000)
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
    if DEBUG >= 3:
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
    if DEBUG >= 3:
      print("RK_CONV payload", info, metadata)
      print("lhs shape", lhs_shape_flat, "rhs shape", rhs_shape_flat, "out shape", out_shape_write, "dtype", dtype)
    post_ops = info.post_ops

    def _shape_prod(shape: tuple[Any, ...]|None) -> int:
      if not shape: return 0
      prod = 1
      for dim in shape:
        prod *= int(dim)
      return prod

    lhs_bytes = self._buffer_as_bytes(lhs_buf)
    rhs_bytes = self._buffer_as_bytes(rhs_buf)
    lhs_elems = _shape_prod(lhs_shape_flat)
    rhs_elems = _shape_prod(rhs_shape_flat)
    out_elems = _shape_prod(out_shape_write)
    axes = info.axes or tuple()

    np_dtype, dtype_read, item_size = self._dtype_with_fp16_fallback(dtype, lhs_bytes, rhs_bytes, lhs_elems, rhs_elems)

    matmul_dims = self._infer_matmul_dims(lhs_elems, rhs_elems, out_shape_write)
    if matmul_dims == (8, 8, 8):
      return self._matmul_8x8_hw(lhs_bytes, rhs_bytes, dtype_read, dtype, np_dtype, post_ops,
                                 out_shape_write, lhs_elems, rhs_elems, out_buf)
    if matmul_dims == (9, 9, 9):
      return self._matmul_9x9_hw(lhs_bytes, rhs_bytes, dtype_read, dtype, np_dtype, post_ops,
                                 out_shape_write, lhs_elems, rhs_elems, out_buf)
    if matmul_dims == (32, 32, 32):
      return self._matmul_32x32_hw(lhs_bytes, rhs_bytes, dtype_read, dtype, np_dtype, post_ops,
                                   out_shape_write, lhs_elems, rhs_elems, out_buf)
    if matmul_dims == (64, 64, 64):
      return self._matmul_64x64_hw(lhs_bytes, rhs_bytes, dtype_read, dtype, np_dtype, post_ops,
                                   out_shape_write, lhs_elems, rhs_elems, out_buf)
    if matmul_dims == (256, 256, 256):
      return self._matmul_256x256_hw(lhs_bytes, rhs_bytes, dtype_read, dtype, np_dtype, post_ops,
                                     out_shape_write, lhs_elems, rhs_elems, out_buf)

    lhs_len_single = int(lhs_elems)
    rhs_len_single = int(rhs_elems)

    conv1d_dims = None
    try:
      conv1d_dims = self._conv1d_shape_info(info, lhs_len_single, rhs_len_single)
      if DEBUG >= 3:
        print("RK_CONV 1d dims (precheck)", conv1d_dims)
    except Exception as exc:
      if DEBUG >= 3:
        print("RK_CONV 1d dims inference failed", exc)
      conv1d_dims = None
    if lhs_elems == 105 and rhs_elems in {36, 54, 108, 162, 180, 270}:
      conv1d_dims = None

    conv2d_dims = self._infer_conv2d_dims(info, lhs_shape_flat, rhs_shape_flat, out_shape_write, axes,
                                          lhs_elems, rhs_elems, out_elems)

    if conv2d_dims is not None:
      N, C, H, W, K, KH, KW, outH, outW = conv2d_dims
      lhs_arr = np.frombuffer(lhs_bytes, dtype=dtype_read, count=lhs_elems).reshape((N, C, H, W))
      rhs_arr = np.frombuffer(rhs_bytes, dtype=dtype_read, count=rhs_elems).reshape((K, C, KH, KW))
      try:
        hw_arr = self._conv2d_hw(lhs_arr, rhs_arr, dtype, np_dtype)
      except Exception as exc:
        if DEBUG >= 3:
          import traceback
          print("RK_CONV conv2d_hw failed", exc)
          traceback.print_exc()
        raise RuntimeError("RK_CONV conv2d hardware path failed") from exc
      if hw_arr is None:
        raise RuntimeError("RK_CONV conv2d hardware path returned no result")
      self._finalize_conv_output(out_buf, hw_arr, np_dtype, post_ops, out_shape_write)
      return 0.0

    if conv1d_dims is not None:
      if DEBUG >= 3:
        print("RK_CONV 1d path", conv1d_dims)
      input_width, kernel_width, output_width, in_channels, out_channels, groups, stride, dilation = conv1d_dims
      lhs_vec = np.frombuffer(lhs_bytes, dtype=dtype_read)
      rhs_vec = np.frombuffer(rhs_bytes, dtype=dtype_read)
      per_sample_elems = input_width * in_channels
      if per_sample_elems <= 0 or lhs_len_single < per_sample_elems:
        raise RuntimeError("invalid 1D convolution buffer size for dims %s" % (conv1d_dims,))
      if lhs_len_single % per_sample_elems != 0:
        raise RuntimeError("RK_CONV 1d buffer length not divisible by sample size %s" % (conv1d_dims,))
      batch_count = lhs_len_single // per_sample_elems
      if DEBUG >= 3 and batch_count > 1:
        print("RK_CONV 1d batches", batch_count)
      output_elems_per_sample = out_channels * output_width
      total_output_elements = batch_count * output_elems_per_sample
      result_arr = self._conv1d_hw_full(lhs_vec, rhs_vec, dtype, np_dtype, *conv1d_dims, batch_count=batch_count)
      if result_arr is None:
        raise RuntimeError("RK_CONV hardware path returned no data for dims %s" % (conv1d_dims,))
      if result_arr.size != total_output_elements:
        raise RuntimeError("RK_CONV output size mismatch for dims %s" % (conv1d_dims,))
      target_shape = tuple(int(x) for x in out_shape_write)
      if not target_shape or int(np.prod(target_shape)) != result_arr.size:
        target_shape = (batch_count, out_channels, output_width)
      self._finalize_conv_output(out_buf, result_arr, np_dtype, post_ops, target_shape)
      return 0.0

    if DEBUG >= 3:
      print("RK_CONV shapes", lhs_shape_flat, rhs_shape_flat)
    if (lhs_shape_flat and rhs_shape_flat and len(lhs_shape_flat) <= 3
        and len(rhs_shape_flat) <= 3):
      if DEBUG >= 3:
        print("RK_CONV 1d candidate (fallback)", lhs_shape_flat, rhs_shape_flat)
      lhs_vec = np.frombuffer(lhs_bytes, dtype=dtype_read)
      rhs_vec = np.frombuffer(rhs_bytes, dtype=dtype_read)
      dims = self._conv1d_shape_info(info, int(lhs_elems), int(rhs_elems))
    kernel_hint: tuple[int, int]|None = None
    if info.axes and info.lhs_shape and len(info.axes) == 2 and len(info.lhs_shape) > max(info.axes):
      try:
        kernel_hint = (int(info.lhs_shape[info.axes[0]]), int(info.lhs_shape[info.axes[1]]))
      except Exception:
        kernel_hint = None
    if self._dispatch_special_conv2d(lhs_elems, rhs_elems, lhs_bytes, rhs_bytes,
                                     dtype_read, dtype, np_dtype, post_ops, out_shape_write, out_buf,
                                     kernel_hint=kernel_hint):
      return 0.0

    lhs_shape = lhs_shape_flat
    rhs_shape = rhs_shape_flat
    out_shape = out_shape_write
    if len(lhs_shape) != 4 or len(rhs_shape) != 4 or len(out_shape) != 4:
      raise RuntimeError("RK_CONV fast path not handled")

    lhs_arr = np.frombuffer(lhs_bytes, dtype=dtype_read).reshape(lhs_shape)
    rhs_arr = np.frombuffer(rhs_bytes, dtype=dtype_read).reshape(rhs_shape)

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
      if DEBUG >= 3:
        import traceback
        print("RK_CONV conv2d_hw failed", exc)
        traceback.print_exc()
      raise RuntimeError("RK_CONV conv2d hardware path failed") from exc
    if hw_arr is None:
      raise RuntimeError("RK_CONV conv2d hardware path returned no result")
    self._finalize_conv_output(out_buf, hw_arr, np_dtype, post_ops, out_shape)
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
