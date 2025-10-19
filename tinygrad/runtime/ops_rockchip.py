# pylint: disable=cell-var-from-loop
# a python uops emulator
# works to test the tensor cores, and all the uops in general
# this is the (living) definition of uops
import array
import ctypes
import functools
import math
import mmap
import os
from typing import Any, TYPE_CHECKING, NamedTuple
import pickle, base64, itertools, time, struct, sys
from tinygrad.dtype import DType, dtypes, ImageDType, PtrDType, truncate
from tinygrad.helpers import all_same, getenv, flatten, get_single_element, mv_address, to_mv
from tinygrad.device import BufferSpec, Compiled, Compiler, Allocator
from tinygrad.codegen.opt import tc
from tinygrad.runtime.ops_cpu import HCQBuffer
from tinygrad.runtime.support.hcq import FileIOInterface, HCQAllocatorBase
from tinygrad.uop.ops import exec_alu, Ops, UOp, GroupOp
from tinygrad.renderer import Renderer
from tinygrad.runtime.autogen import rockchip as rk

import sys, numpy as np
np.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=False)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

NPU_CBUF_BANK_SIZE = 32768
NPU_CBUF_BANKS = 12
MATMUL_CHANNEL_ALIGN = 32
MATMUL_THREAD_CHUNK = 8
PC_ENABLE = 0x01
PC_ENABLE_CNA = 0x04
PC_ENABLE_DPU = 0x08
BLOCK_PC = 0x0100
BLOCK_CNA = 0x0200
BLOCK_CORE = 0x0800
BLOCK_DPU = 0x1000
PC_OP_01 = 0x01
PC_OP_40 = 0x40
PC_OP_ENABLE = 0x80
OP_REG_PC = BLOCK_PC | PC_OP_01
OP_REG_CNA = BLOCK_CNA | PC_OP_01
OP_REG_CORE = BLOCK_CORE | PC_OP_01
OP_REG_DPU = BLOCK_DPU | PC_OP_01
OP_40 = PC_OP_40 | PC_OP_01
OP_ENABLE = PC_OP_ENABLE | PC_OP_01
OP_NONE = 0x0
DIRECT_CONVOLUTION = 0
PRECISION_INT8 = 0
PRECISION_FLOAT16 = 2
PRECISION_FLOAT32 = 5
REG_CORE_3030 = 0x00003030
REG_DPU_40C4 = 0x000040C4


class MatmulParams(ctypes.Structure):
  _fields_ = [
    ("m", ctypes.c_uint16),
    ("k", ctypes.c_uint16),
    ("n", ctypes.c_uint16),
    ("_pad0", ctypes.c_uint16),
    ("input_dma", ctypes.c_uint32),
    ("weights_dma", ctypes.c_uint32),
    ("output_dma", ctypes.c_uint32),
    ("tasks", ctypes.POINTER(ctypes.c_uint64)),
    ("fp32tofp16", ctypes.c_uint8),
  ]

class ConvConfig(NamedTuple):
  ndim: int
  batch: int
  groups: int
  cin: int
  cout: int
  cin_per_group: int
  cout_per_group: int
  out_shape: tuple[int, ...]
  kernel_shape: tuple[int, ...]
  in_shape_tail: tuple[int, ...]


def _align_up(value:int, align:int) -> int:
  return (value + align - 1) // align * align

def _divisors(n:int) -> list[int]:
  n = abs(n)
  if n == 0: return []
  divs:set[int] = set()
  for i in range(1, int(math.isqrt(n)) + 1):
    if n % i == 0:
      divs.add(i)
      divs.add(n//i)
  return sorted(divs)

def _weight_fp16_index(channels:int, kernel_idx:int, channel_idx:int) -> int:
  k = kernel_idx + 1
  c = channel_idx + 1
  kpg = (k - 1) // 16
  cpg = (c - 1) // 32
  idx = ((cpg * 32) * 16) + (kpg * 16 * channels)
  idx += ((c - 1) % 32) + (((k - 1) % 16) * 32)
  return idx

def _feature_fp16_index(channels:int, height:int, channel_idx:int, row_idx:int, chunk:int=MATMUL_THREAD_CHUNK) -> int:
  c = channel_idx + 1
  h = row_idx + 1
  plane = (c - 1) // chunk
  src = plane * height * chunk
  offset = (c - 1) % chunk
  return src + chunk * (h - 1) + offset

def _npuop(op:int, value:int, reg:int) -> int:
  return ((op & 0xffff) << 48) | ((value & 0xffffffff) << 16) | (reg & 0xffff)

def _gen_matmul_fp16(params:MatmulParams) -> int:
  m = int(params.m)
  k = int(params.k)
  n = int(params.n)
  input_dma = int(params.input_dma)
  weights_dma = int(params.weights_dma)
  output_dma = int(params.output_dma)
  fp32tofp16 = int(params.fp32tofp16 & 0x1)

  datain_width = 1
  datain_height = m
  datain_channel = k
  dataout_width = 1
  dataout_height = m
  weight_width = 1
  weight_height = 1
  weight_kernels = n

  weight_bytes_per_kernel = weight_width * weight_height * datain_channel * ctypes.sizeof(ctypes.c_uint16)
  weight_bytes = weight_bytes_per_kernel * weight_kernels
  fd_bytes = datain_width * datain_height * datain_channel * ctypes.sizeof(ctypes.c_uint16)

  fd_banks = (fd_bytes + NPU_CBUF_BANK_SIZE - 1) // NPU_CBUF_BANK_SIZE
  weight_banks = (weight_bytes + NPU_CBUF_BANK_SIZE - 1) // NPU_CBUF_BANK_SIZE
  if fd_banks > NPU_CBUF_BANKS - 1:
    return -1
  if weight_bytes_per_kernel <= NPU_CBUF_BANK_SIZE:
    weight_banks = NPU_CBUF_BANKS - fd_banks
  else:
    return -2

  data_entries = (datain_width * datain_channel + 31) // 32
  line_stride = datain_width * 4
  surf_stride = line_stride * ((datain_height // 4) - 1)
  if surf_stride < 0:
    surf_stride += 1

  cna = {
    "proc_precision": PRECISION_FLOAT16,
    "in_precision": PRECISION_FLOAT16,
    "conv_mode": DIRECT_CONVOLUTION,
    "kernel_groups": 0,
    "feature_grains": m + 1,
    "conv_x_stride": 1,
    "conv_y_stride": 1,
    "datain_width": datain_width,
    "datain_height": datain_height,
    "datain_channel": datain_channel,
    "dataout_width": dataout_width,
    "dataout_height": dataout_height,
    "dataout_atomics": dataout_width * dataout_height,
    "weight_width": weight_width,
    "weight_height": weight_height,
    "weight_kernels": weight_kernels,
    "weight_bytes_per_kernel": weight_bytes_per_kernel,
    "weight_bytes": weight_bytes,
    "weight_bank": weight_banks,
    "data_bank": fd_banks,
    "data_entries": data_entries,
    "data_sign": 0x1,
    "cvt_type": 0x1,
    "cvt_bypass": 0x1,
    "cvt_scale0": 0x1,
    "cvt_scale1": 0x1,
    "cvt_scale2": 0x1,
    "cvt_scale3": 0x1,
    "fc_skip_en": 0,
    "data_offset": 0,
    "pad_left": 0,
    "pad_top": 0,
    "feature_base_addr": input_dma,
    "weight_offset": 0,
    "weight_burst_len": 0xf,
    "data_burst_len": 0xf,
    "line_stride": line_stride,
    "surf_stride": surf_stride,
    "dma_width": datain_width,
    "dma_height": datain_height,
    "dma_channel": datain_channel,
    "decompress_addr0": weights_dma,
  }

  core = {
    "proc_precision": PRECISION_FLOAT16,
    "qd_en": 1,
    "dataout_height": max(dataout_height - 1, 0),
    "dataout_width": max(dataout_width - 1, 0),
    "dataout_channel": max(weight_kernels - 1, 0),
  }

  dst_surf_stride = dataout_height * dataout_width
  convert_out = PRECISION_FLOAT16 if fp32tofp16 else PRECISION_FLOAT32
  size_e_val = 1 if fp32tofp16 else 3
  surf_add_scale = 2 if fp32tofp16 else 4

  dpu = {
    "burst_len": 0xf,
    "conv_mode": DIRECT_CONVOLUTION,
    "output_mode": 0x2,
    "flying_mode": 0x0,
    "out_precision": convert_out,
    "in_precision": PRECISION_FLOAT16,
    "proc_precision": PRECISION_FLOAT16,
    "dst_base_addr": output_dma,
    "dst_surf_stride": dst_surf_stride,
    "width": core["dataout_width"],
    "height": core["dataout_height"],
    "channel": core["dataout_channel"],
    "bs_bypass": 1,
    "bs_alu_bypass": 1,
    "bs_mul_bypass": 1,
    "bs_relu_bypass": 1,
    "bn_bypass": 1,
    "bn_alu_bypass": 1,
    "bn_mul_bypass": 1,
    "bn_relu_bypass": 1,
    "ew_bypass": 1,
    "ew_op_bypass": 1,
    "ew_lut_bypass": 1,
    "ew_op_cvt_bypass": 1,
    "ew_relu_bypass": 1,
    "fp32tofp16_en": fp32tofp16,
    "out_cvt_scale": 1,
    "size_e_2": size_e_val,
    "size_e_1": size_e_val,
    "size_e_0": size_e_val,
    "od_bypass": 1,
    "width_wdma": core["dataout_width"],
    "height_wdma": core["dataout_height"],
    "channel_wdma": core["dataout_channel"],
    "surf_add": dst_surf_stride * surf_add_scale,
  }

  tasks = params.tasks
  if not tasks:
    return -3

  ops:list[int] = []
  ops.append(_npuop(OP_REG_DPU, 0xE, rk.REG_DPU_S_POINTER))
  value = ((cna["proc_precision"] & 0x7) << 7) | ((cna["in_precision"] & 0x7) << 4) | (cna["conv_mode"] & 0xf)
  ops.append(_npuop(OP_REG_CNA, value, rk.REG_CNA_CONV_CON1))
  value = ((cna["kernel_groups"] & 0xFF) << 16) | ((cna["feature_grains"] & 0x3FF) << 4)
  ops.append(_npuop(OP_REG_CNA, value, rk.REG_CNA_CONV_CON2))
  value = ((cna["conv_y_stride"] & 0x7) << 3) | (cna["conv_x_stride"] & 0x7)
  ops.append(_npuop(OP_REG_CNA, value, rk.REG_CNA_CONV_CON3))
  value = ((cna["datain_width"] & 0x7FF) << 16) | (cna["datain_height"] & 0x7FF)
  ops.append(_npuop(OP_REG_CNA, value, rk.REG_CNA_DATA_SIZE0))
  value = (((cna["datain_channel"] - 1) & 0xFFFF) << 16) | (cna["datain_channel"] & 0xFFFF)
  ops.append(_npuop(OP_REG_CNA, value, rk.REG_CNA_DATA_SIZE1))
  value = cna["dataout_width"] & 0x7FF
  ops.append(_npuop(OP_REG_CNA, value, rk.REG_CNA_DATA_SIZE2))
  value = cna["dataout_atomics"] & 0x3FFFF
  ops.append(_npuop(OP_REG_CNA, value, rk.REG_CNA_DATA_SIZE3))
  ops.append(_npuop(OP_REG_CNA, cna["weight_bytes"], rk.REG_CNA_WEIGHT_SIZE0))
  value = cna["weight_bytes_per_kernel"] & 0x7FFFF
  ops.append(_npuop(OP_REG_CNA, value, rk.REG_CNA_WEIGHT_SIZE1))
  value = ((cna["weight_width"] & 0x1F) << 24) | ((cna["weight_height"] & 0x1F) << 16) | (cna["weight_kernels"] & 0x3FFF)
  ops.append(_npuop(OP_REG_CNA, value, rk.REG_CNA_WEIGHT_SIZE2))
  value = ((cna["weight_bank"] & 0xF) << 4) | (cna["data_bank"] & 0xF)
  ops.append(_npuop(OP_REG_CNA, value, rk.REG_CNA_CBUF_CON0))
  ops.append(_npuop(OP_REG_CNA, cna["data_entries"] & 0x1FFF, rk.REG_CNA_CBUF_CON1))
  value = ((cna["data_sign"] & 0x1) << 3) | ((cna["cvt_type"] & 0x1) << 1) | (cna["cvt_bypass"] & 0x1)
  ops.append(_npuop(OP_REG_CNA, value, rk.REG_CNA_CVT_CON0))
  ops.append(_npuop(OP_REG_CNA, (cna["cvt_scale0"] & 0xFFFF) << 16, rk.REG_CNA_CVT_CON1))
  ops.append(_npuop(OP_REG_CNA, (cna["cvt_scale1"] & 0xFFFF) << 16, rk.REG_CNA_CVT_CON2))
  ops.append(_npuop(OP_REG_CNA, (cna["cvt_scale2"] & 0xFFFF) << 16, rk.REG_CNA_CVT_CON3))
  ops.append(_npuop(OP_REG_CNA, (cna["cvt_scale3"] & 0xFFFF) << 16, rk.REG_CNA_CVT_CON4))
  ops.append(_npuop(OP_REG_CNA, cna["fc_skip_en"] & 0x1, rk.REG_CNA_FC_CON0))
  ops.append(_npuop(OP_REG_CNA, cna["data_offset"] & 0x1FFFF, rk.REG_CNA_FC_CON1))
  value = ((cna["pad_left"] & 0xF) << 4) | (cna["pad_top"] & 0xF)
  ops.append(_npuop(OP_REG_CNA, value, rk.REG_CNA_PAD_CON0))
  ops.append(_npuop(OP_REG_CNA, cna["feature_base_addr"], rk.REG_CNA_FEATURE_DATA_ADDR))
  ops.append(_npuop(OP_REG_CNA, cna["weight_offset"] & 0x1FFFF, rk.REG_CNA_FC_CON2))
  value = ((cna["weight_burst_len"] & 0xF) << 16) | (cna["data_burst_len"] & 0xF)
  ops.append(_npuop(OP_REG_CNA, value, rk.REG_CNA_DMA_CON0))
  ops.append(_npuop(OP_REG_CNA, cna["line_stride"] & 0xFFFFFFF, rk.REG_CNA_DMA_CON1))
  ops.append(_npuop(OP_REG_CNA, cna["surf_stride"] & 0xFFFFFFF, rk.REG_CNA_DMA_CON2))
  value = ((cna["dma_width"] & 0x7FF) << 16) | (cna["dma_height"] & 0x7FF)
  ops.append(_npuop(OP_REG_CNA, value, rk.REG_CNA_FC_DATA_SIZE0))
  ops.append(_npuop(OP_REG_CNA, cna["dma_channel"] & 0xFFFF, rk.REG_CNA_FC_DATA_SIZE1))
  ops.extend([
    _npuop(OP_REG_CNA, 0x0, rk.REG_CNA_DCOMP_CTRL),
    _npuop(OP_REG_CNA, 0x0, rk.REG_CNA_DCOMP_REGNUM),
    _npuop(OP_REG_CNA, cna["decompress_addr0"], rk.REG_CNA_DCOMP_ADDR0),
  ])
  for idx in range(16):
    reg = getattr(rk, f"REG_CNA_DCOMP_AMOUNT{idx}")
    ops.append(_npuop(OP_REG_CNA, 0x0, reg))
  ops.append(_npuop(OP_REG_CNA, 0x0, rk.REG_CNA_CVT_CON5))
  ops.append(_npuop(OP_REG_CNA, 0x0, rk.REG_CNA_PAD_CON1))

  value = ((core["proc_precision"] & 0x7) << 8) | (core["qd_en"] & 0x1)
  ops.append(_npuop(OP_REG_CORE, value, rk.REG_CORE_MISC_CFG))
  value = ((core["dataout_height"] & 0xFFFF) << 16) | (core["dataout_width"] & 0xFFFF)
  ops.append(_npuop(OP_REG_CORE, value, rk.REG_CORE_DATAOUT_SIZE_0))
  ops.append(_npuop(OP_REG_CORE, core["dataout_channel"] & 0xFFFF, rk.REG_CORE_DATAOUT_SIZE_1))
  ops.append(_npuop(OP_REG_CORE, 0x0, rk.REG_CORE_CLIP_TRUNCATE))
  ops.append(_npuop(OP_REG_CORE, 0x0, REG_CORE_3030))

  value = ((dpu["burst_len"] & 0xF) << 5) | ((dpu["conv_mode"] & 0x3) << 3) | ((dpu["output_mode"] & 0x3) << 1) | (dpu["flying_mode"] & 0x1)
  ops.append(_npuop(OP_REG_DPU, value, rk.REG_DPU_FEATURE_MODE_CFG))
  value = ((dpu["out_precision"] & 0x7) << 29) | ((dpu["in_precision"] & 0x7) << 26) | (dpu["proc_precision"] & 0x7)
  ops.append(_npuop(OP_REG_DPU, value, rk.REG_DPU_DATA_FORMAT))
  ops.append(_npuop(OP_REG_DPU, 0x0, rk.REG_DPU_OFFSET_PEND))
  ops.append(_npuop(OP_REG_DPU, dpu["dst_base_addr"], rk.REG_DPU_DST_BASE_ADDR))
  ops.append(_npuop(OP_REG_DPU, (dpu["dst_surf_stride"] & 0xFFFFFFF) << 4, rk.REG_DPU_DST_SURF_STRIDE))
  ops.append(_npuop(OP_REG_DPU, dpu["width"] & 0x1FFF, rk.REG_DPU_DATA_CUBE_WIDTH))
  ops.append(_npuop(OP_REG_DPU, dpu["height"] & 0x1FFF, rk.REG_DPU_DATA_CUBE_HEIGHT))
  ops.append(_npuop(OP_REG_DPU, 0x0, rk.REG_DPU_DATA_CUBE_NOTCH_ADDR))
  value = ((dpu["channel"] & 0x1FFF) << 16) | (dpu["channel"] & 0x1FFF)
  ops.append(_npuop(OP_REG_DPU, value, rk.REG_DPU_DATA_CUBE_CHANNEL))
  value = ((dpu["bs_relu_bypass"] & 0x1) << 6) | ((dpu["bs_mul_bypass"] & 0x1) << 4) | ((dpu["bs_alu_bypass"] & 0x1) << 1) | (dpu["bs_bypass"] & 0x1)
  ops.append(_npuop(OP_REG_DPU, value, rk.REG_DPU_BS_CFG))
  for reg in (rk.REG_DPU_BS_ALU_CFG, rk.REG_DPU_BS_MUL_CFG, rk.REG_DPU_BS_RELUX_CMP_VALUE):
    ops.append(_npuop(OP_REG_DPU, 0x0, reg))
  value = ((dpu["size_e_2"] & 0x7) << 8) | ((dpu["size_e_1"] & 0x7) << 5) | ((dpu["size_e_0"] & 0x7) << 2) | ((dpu["od_bypass"] & 0x1) << 1)
  ops.append(_npuop(OP_REG_DPU, value, rk.REG_DPU_BS_OW_CFG))
  ops.append(_npuop(OP_REG_DPU, 0x0, rk.REG_DPU_BS_OW_OP))
  ops.append(_npuop(OP_REG_DPU, dpu["channel_wdma"] & 0x1FFF, rk.REG_DPU_WDMA_SIZE_0))
  value = ((dpu["height_wdma"] & 0x1FFF) << 16) | (dpu["width_wdma"] & 0x1FFF)
  ops.append(_npuop(OP_REG_DPU, value, rk.REG_DPU_WDMA_SIZE_1))
  value = ((dpu["bn_relu_bypass"] & 0x1) << 6) | ((dpu["bn_mul_bypass"] & 0x1) << 4) | ((dpu["bn_alu_bypass"] & 0x1) << 1) | (dpu["bn_bypass"] & 0x1)
  ops.append(_npuop(OP_REG_DPU, value, rk.REG_DPU_BN_CFG))
  for reg in (rk.REG_DPU_BN_ALU_CFG, rk.REG_DPU_BN_MUL_CFG, rk.REG_DPU_BN_RELUX_CMP_VALUE):
    ops.append(_npuop(OP_REG_DPU, 0x0, reg))
  value = ((dpu["ew_relu_bypass"] & 0x1) << 9) | ((dpu["ew_op_cvt_bypass"] & 0x1) << 8) | ((dpu["ew_lut_bypass"] & 0x1) << 7) | ((dpu["ew_op_bypass"] & 0x1) << 1) | (dpu["ew_bypass"] & 0x1)
  ops.append(_npuop(OP_REG_DPU, value, rk.REG_DPU_EW_CFG))
  ops.append(_npuop(OP_REG_DPU, 0x0, rk.REG_DPU_EW_CVT_OFFSET_VALUE))
  ops.append(_npuop(OP_REG_DPU, 0x1, rk.REG_DPU_EW_CVT_SCALE_VALUE))
  ops.append(_npuop(OP_REG_DPU, 0x0, rk.REG_DPU_EW_RELUX_CMP_VALUE))
  ops.append(_npuop(OP_REG_DPU, 0x0, rk.REG_DPU_OUT_CVT_OFFSET))
  value = ((dpu["fp32tofp16_en"] & 0x1) << 16) | (dpu["out_cvt_scale"] & 0xFFFF)
  ops.append(_npuop(OP_REG_DPU, value, rk.REG_DPU_OUT_CVT_SCALE))
  ops.append(_npuop(OP_REG_DPU, 0x0, rk.REG_DPU_OUT_CVT_SHIFT))
  for reg in (
    rk.REG_DPU_EW_OP_VALUE_0, rk.REG_DPU_EW_OP_VALUE_1, rk.REG_DPU_EW_OP_VALUE_2,
    rk.REG_DPU_EW_OP_VALUE_3, rk.REG_DPU_EW_OP_VALUE_4, rk.REG_DPU_EW_OP_VALUE_5,
    rk.REG_DPU_EW_OP_VALUE_6, rk.REG_DPU_EW_OP_VALUE_7):
    ops.append(_npuop(OP_REG_DPU, 0x0, reg))
  ops.append(_npuop(OP_REG_DPU, (dpu["surf_add"] & 0xFFFFFFF) << 4, rk.REG_DPU_SURFACE_ADD))
  ops.append(_npuop(OP_REG_DPU, 0x0, REG_DPU_40C4))
  for reg in (
    rk.REG_DPU_LUT_ACCESS_CFG, rk.REG_DPU_LUT_ACCESS_DATA, rk.REG_DPU_LUT_CFG, rk.REG_DPU_LUT_INFO,
    rk.REG_DPU_LUT_LE_START, rk.REG_DPU_LUT_LE_END, rk.REG_DPU_LUT_LO_START, rk.REG_DPU_LUT_LO_END,
    rk.REG_DPU_LUT_LE_SLOPE_SCALE, rk.REG_DPU_LUT_LE_SLOPE_SHIFT,
    rk.REG_DPU_LUT_LO_SLOPE_SCALE, rk.REG_DPU_LUT_LO_SLOPE_SHIFT):
    ops.append(_npuop(OP_REG_DPU, 0x0, reg))

  ops.append(_npuop(OP_NONE, 0x0, 0x0))
  ops.append(_npuop(OP_REG_PC, 0x0, rk.REG_PC_REGISTER_AMOUNTS))
  ops.append(_npuop(OP_40, 0x0, 0x0))
  ops.append(_npuop(OP_ENABLE, PC_ENABLE_DPU | PC_ENABLE_CNA | PC_ENABLE, rk.REG_PC_OPERATION_ENABLE))

  for idx, val in enumerate(ops):
    tasks[idx] = ctypes.c_uint64(val)
  return 0


def storage_fmt_for_dtype(dtype: DType):
  if dtype in (dtypes.bfloat16, dtypes.float16): return 'H'
  return dtype.fmt

def to_storage_scalar(x, dtype: DType):
  if dtype == dtypes.bfloat16: return (struct.unpack('I', struct.pack('f', float_to_bf16(x)))[0] >> 16) & 0xFFFF
  if dtype == dtypes.float16: return int(np.float16(x).view(np.uint16))
  return x

def from_storage_scalar(x, dtype: DType):
  if dtype == dtypes.bfloat16: return struct.unpack('f', struct.pack('I', (x & 0xFFFF) << 16))[0]
  if dtype == dtypes.float16: return float(np.uint16(x).view(np.float16))
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

  def render(self, uops:list[UOp]) -> str:
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
    is_float_add = (op == Ops.ADD and dtype == dtypes.float)
    in_dtype = dtypes.float16 if is_float_add else dtype
    proc_dtype = dtypes.float16 if is_float_add else dtype
    out_dtype = dtypes.float32 if is_float_add else dtype
    ew_op_cvt_bypass = 1 if is_float_add else 0

    self.emit_raw(rk.DPU, rk.REG_DPU_DATA_FORMAT,
      self.reg(self.get_precision(out_dtype), rk.DPU_DATA_FORMAT_OUT_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_OUT_PRECISION__MASK) |
      self.reg(self.get_precision(in_dtype), rk.DPU_DATA_FORMAT_IN_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_IN_PRECISION__MASK) |
      self.reg(self.get_precision(proc_dtype), rk.DPU_DATA_FORMAT_PROC_PRECISION__SHIFT, rk.DPU_DATA_FORMAT_PROC_PRECISION__MASK))

    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
      self.reg(self.get_precision(in_dtype), rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_IN_PRECISION__MASK) |
      self.reg(15, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN__MASK) |
      self.reg(0, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE__MASK) |
      self.reg(self.get_precision(proc_dtype), rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_PROC_PRECISION__MASK) |
      self.reg(0, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE__MASK) |
      self.reg(self.get_is_fp16(in_dtype), rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_FP16TOFP32_EN__MASK) |
      self.reg(0, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE__MASK) |
      self.reg(1, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__SHIFT, rk.DPU_RDMA_RDMA_FEATURE_MODE_CFG_FLYING_MODE__MASK))

    fp32_to_fp16_en = 1 if out_dtype == dtypes.float16 else 0
    out_cvt_scale = 0 if out_dtype == dtypes.float32 else 1
    self.emit_raw(rk.DPU, rk.REG_DPU_OUT_CVT_SCALE, 
      self.reg(fp32_to_fp16_en, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__SHIFT, rk.DPU_OUT_CVT_SCALE_FP32TOFP16_EN__MASK) |
      self.reg(out_cvt_scale, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__SHIFT, rk.DPU_OUT_CVT_SCALE_OUT_CVT_SCALE__MASK));

    self.emit_raw(rk.DPU_RDMA, rk.REG_DPU_RDMA_RDMA_ERDMA_CFG,
      self.reg(1, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE__MASK) |
      self.reg(self.get_edata_size(in_dtype), rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__SHIFT, rk.DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE__MASK))
    
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
        self.reg(self.get_edata_size(in_dtype), rk.DPU_EW_CFG_EDATA_SIZE__SHIFT, rk.DPU_EW_CFG_EDATA_SIZE__MASK) |
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
        self.reg(self.get_edata_size(in_dtype), rk.DPU_EW_CFG_EDATA_SIZE__SHIFT, rk.DPU_EW_CFG_EDATA_SIZE__MASK) |
        self.reg(0, rk.DPU_EW_CFG_EW_EQUAL_EN__SHIFT, rk.DPU_EW_CFG_EW_EQUAL_EN__MASK) |
        self.reg(0, rk.DPU_EW_CFG_EW_BINARY_EN__SHIFT, rk.DPU_EW_CFG_EW_BINARY_EN__MASK) |
        self.reg(self.code_for_op[op], rk.DPU_EW_CFG_EW_ALU_ALGO__SHIFT, rk.DPU_EW_CFG_EW_ALU_ALGO__MASK) |
        self.reg(0, rk.DPU_EW_CFG_EW_RELUX_EN__SHIFT, rk.DPU_EW_CFG_EW_RELUX_EN__MASK) |
        self.reg(1, rk.DPU_EW_CFG_EW_RELU_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_RELU_BYPASS__MASK) |
        self.reg(ew_op_cvt_bypass, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__SHIFT, rk.DPU_EW_CFG_EW_OP_CVT_BYPASS__MASK) |
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
            timeout=20000,
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
    self.uops: list[tuple[Ops, DType|None, list[int], Any]] = pickle.loads(lib)
    self.device = dev
    self.q = []
    self.code_for_op = RockchipRenderer.code_for_op
    self.name = name
    self.matmul_bufs: dict[tuple[int,int,int], tuple[HCQBuffer, HCQBuffer, HCQBuffer]] = {}

  def _maybe_run_matmul(self, bufs:tuple[Any, ...]) -> bool:
    if self._maybe_run_conv(bufs):
      return True
    name = getattr(self, "name", "")
    if not name.startswith("r_"):
      return False
    parts = name.split('_')
    if len(parts) < 4 or not all(part.isdigit() for part in parts[1:4]):
      return False
    if len(bufs) < 3:
      return False
    M, K, N = (int(parts[1]), int(parts[2]), int(parts[3]))
    try:
      self._run_matmul_conv(bufs, M, K, N)
      if getenv("DEBUG"):
        print("matmul handled", name, M, K, N)
      return True
    except Exception as exc:
      if getenv("DEBUG"):
        print("matmul conv fallback", exc)
      return False

  def _run_matmul_conv(self, bufs:tuple[Any, ...], M:int, K:int, N:int) -> None:
    try:
      out_buf, a_buf, b_buf = bufs[:3]
      a = np.frombuffer(a_buf, dtype=np.float16, count=M*K).reshape(M, K)
      b = np.frombuffer(b_buf, dtype=np.float16, count=K*N).reshape(K, N)
    except Exception as exc:
      if getenv("DEBUG"):
        print("matmul buffer prep failed", exc)
      raise

    Mpad = max(_align_up(M, MATMUL_CHANNEL_ALIGN), MATMUL_CHANNEL_ALIGN)
    Kpad = max(_align_up(K, MATMUL_CHANNEL_ALIGN), MATMUL_CHANNEL_ALIGN)
    Npad = max(_align_up(N, MATMUL_CHANNEL_ALIGN), MATMUL_CHANNEL_ALIGN)

    feature_arr = np.zeros((Mpad * Kpad,), dtype=np.float16)
    for m in range(M):
      for k in range(K):
        feature_arr[_feature_fp16_index(Kpad, Mpad, k, m)] = a[m, k]

    weight_arr = np.zeros((Npad * Kpad,), dtype=np.float16)
    for n in range(N):
      for k in range(K):
        weight_arr[_weight_fp16_index(Kpad, n, k)] = b[k, n]

    key = (Mpad, Kpad, Npad)
    if key not in self.matmul_bufs:
      input_buf = self.device._gpu_alloc(feature_arr.nbytes, 0)
      weight_buf = self.device._gpu_alloc(weight_arr.nbytes, 0)
      output_buf = self.device._gpu_alloc(Mpad * Npad * ctypes.sizeof(ctypes.c_float), 0)
      self.matmul_bufs[key] = (input_buf, weight_buf, output_buf)
    else:
      input_buf, weight_buf, output_buf = self.matmul_bufs[key]

    ctypes.memmove(input_buf.va_addr, feature_arr.tobytes(), feature_arr.nbytes)
    ctypes.memmove(weight_buf.va_addr, weight_arr.tobytes(), weight_arr.nbytes)
    if getenv("DEBUG") >= 3:
      print("matmul dma", hex(input_buf.meta.dma_addr), hex(weight_buf.meta.dma_addr), hex(output_buf.meta.dma_addr))

    q_vals = self._build_matmul_queue(Mpad, Kpad, Npad,
      input_buf.meta.dma_addr, weight_buf.meta.dma_addr, output_buf.meta.dma_addr)
    self._submit_queue(q_vals, op_idx=0, enable_mask=0xd)

    out_bytes = ctypes.create_string_buffer(Mpad * Npad * ctypes.sizeof(ctypes.c_float))
    ctypes.memmove(out_bytes, output_buf.va_addr, out_bytes._length_)
    out_mat = np.frombuffer(out_bytes, dtype=np.float32).reshape(Mpad, Npad)
    if getenv("DEBUG") >= 3:
      coords = [(i, j, out_mat[i, j]) for i in range(Mpad) for j in range(Npad) if not np.isclose(out_mat[i, j], 0)]
      print("raw matmul nonzero coords", coords[:32])
    if getenv("ROCKCHIP_DUMP_RAW"):
      np.save("/tmp/rockchip_matmul_raw.npy", out_mat)
    decoded = np.zeros((Mpad, Npad), dtype=np.float32)
    # Rockchip stores the matmul tile as interleaved grids: ri mod 4 picks a block of (Mpad//4) rows
    # and ri // 4 selects a group of 4 columns. Within each tile ci // 4 steps through rows and ci % 4
    # picks the column inside the group.
    row_tile = Mpad // 4
    for ri in range(Mpad):
      for ci in range(Npad):
        val = out_mat[ri, ci]
        if val == 0: continue
        row_idx = (ri % 4) * row_tile + (ci // 4)
        col_idx = (ri // 4) * 4 + (ci % 4)
        if row_idx < Mpad and col_idx < Npad:
          decoded[row_idx, col_idx] = val
    trimmed = decoded[:M, :N].astype(np.float16).reshape(-1)
    out_bytes = trimmed.tobytes()
    if isinstance(out_buf, HCQBuffer):
      ctypes.memmove(out_buf.va_addr, out_bytes, len(out_bytes))
    else:
      mv = out_buf if isinstance(out_buf, memoryview) else memoryview(out_buf)
      mv[:len(out_bytes)] = out_bytes
  def _parse_conv1d_from_name(self, parts:list[int], buf_sizes:list[int]) -> ConvConfig|None:
    output_elems = buf_sizes[0] // 2
    input_elems = buf_sizes[1] // 2
    weight_elems = buf_sizes[2] // 2
    configs:list[ConvConfig] = []

    def add_config(batch:int, groups:int, cin:int, cout:int, out_len:int, k_len:int):
      if batch <= 0 or groups <= 0 or cin <= 0 or cout <= 0 or out_len <= 0 or k_len <= 0: return
      lin = out_len + k_len - 1
      if lin <= 0: return
      if output_elems != batch * cout * out_len: return
      if input_elems != batch * cin * lin: return
      if cin % groups != 0: return
      cin_pg = cin // groups
      if weight_elems != cout * cin_pg * k_len: return
      configs.append(ConvConfig(
        ndim=1,
        batch=batch,
        groups=groups,
        cin=cin,
        cout=cout,
        cin_per_group=cin_pg,
        cout_per_group=cout // groups,
        out_shape=(out_len,),
        kernel_shape=(k_len,),
        in_shape_tail=(lin,),
      ))

    if len(parts) == 3:
      cout, out_len, val = parts
      denom = cout * out_len
      if denom and output_elems % denom == 0:
        batch = output_elems // denom
        k_len = val
        lin = out_len + k_len - 1
        if lin and input_elems % (batch * lin) == 0:
          cin = input_elems // (batch * lin)
          add_config(batch, 1, cin, cout, out_len, k_len)
      cin_candidate = val
      if cin_candidate > 0 and weight_elems % (cout * cin_candidate) == 0:
        k_len = weight_elems // (cout * cin_candidate)
        denom = cout * out_len
        if denom and output_elems % denom == 0:
          batch = output_elems // denom
          lin = out_len + k_len - 1
          if lin and input_elems == batch * cin_candidate * lin:
            add_config(batch, 1, cin_candidate, cout, out_len, k_len)
    if len(parts) == 4:
      cout, out_len, cin, k_len = parts
      denom = cout * out_len
      if denom and output_elems % denom == 0 and input_elems == (output_elems // denom) * cin * (out_len + k_len - 1):
        batch = output_elems // denom
        add_config(batch, 1, cin, cout, out_len, k_len)
      batch, cout, out_len, k_len = parts
      lin = out_len + k_len - 1
      if lin and input_elems % (batch * lin) == 0:
        cin = input_elems // (batch * lin)
        add_config(batch, 1, cin, cout, out_len, k_len)
    if len(parts) == 5:
      batch, cout, out_len, cin, k_len = parts
      add_config(batch, 1, cin, cout, out_len, k_len)
      batch, groups, cout_pg, out_len, k_len = parts
      cout = cout_pg * groups
      lin = out_len + k_len - 1
      if lin and input_elems % (batch * lin) == 0:
        cin = input_elems // (batch * lin)
        add_config(batch, groups, cin, cout, out_len, k_len)
    if len(parts) == 6:
      batch, groups, cout_pg, out_len, cin_pg, k_len = parts
      cout = cout_pg * groups
      cin = cin_pg * groups
      add_config(batch, groups, cin, cout, out_len, k_len)

    if not configs: return None
    return max(configs, key=lambda cfg: (cfg.cin, -cfg.groups, cfg.cout))

  def _parse_conv2d_from_name(self, parts:list[int], buf_sizes:list[int]) -> ConvConfig|None:
    output_elems = buf_sizes[0] // 2
    input_elems = buf_sizes[1] // 2
    weight_elems = buf_sizes[2] // 2
    configs:list[ConvConfig] = []

    def add_config(batch:int, groups:int, cin:int, cout:int, out_h:int, out_w:int, k_h:int, k_w:int):
      if batch <= 0 or groups <= 0 or cin <= 0 or cout <= 0: return
      if out_h <= 0 or out_w <= 0 or k_h <= 0 or k_w <= 0: return
      hin = out_h + k_h - 1
      win = out_w + k_w - 1
      if hin <= 0 or win <= 0: return
      if output_elems != batch * cout * out_h * out_w: return
      if input_elems != batch * cin * hin * win: return
      if cin % groups != 0: return
      cin_pg = cin // groups
      if weight_elems != cout * cin_pg * k_h * k_w: return
      configs.append(ConvConfig(
        ndim=2,
        batch=batch,
        groups=groups,
        cin=cin,
        cout=cout,
        cin_per_group=cin_pg,
        cout_per_group=cout // groups,
        out_shape=(out_h, out_w),
        kernel_shape=(k_h, k_w),
        in_shape_tail=(hin, win),
      ))

    n = len(parts)
    if n == 5:
      cout, out_h, out_w, k_h, k_w = parts
      denom = cout * out_h * out_w
      if denom and output_elems % denom == 0:
        batch = output_elems // denom
        hin = out_h + k_h - 1
        win = out_w + k_w - 1
        if hin and win and input_elems % (batch * hin * win) == 0:
          cin = input_elems // (batch * hin * win)
          add_config(batch, 1, cin, cout, out_h, out_w, k_h, k_w)
    if n == 6:
      cout, out_h, out_w, cin, k_h, k_w = parts
      denom = cout * out_h * out_w
      if denom and output_elems % denom == 0:
        batch = output_elems // denom
        add_config(batch, 1, cin, cout, out_h, out_w, k_h, k_w)
      groups, cout_pg, out_h, out_w, k_h, k_w = parts
      cout = cout_pg * groups
      denom = cout * out_h * out_w
      if denom and output_elems % denom == 0:
        batch = output_elems // denom
        hin = out_h + k_h - 1
        win = out_w + k_w - 1
        if hin and win and input_elems % (batch * hin * win) == 0:
          cin = input_elems // (batch * hin * win)
          add_config(batch, groups, cin, cout, out_h, out_w, k_h, k_w)
      batch, cout, out_h, out_w, k_h, k_w = parts
      hin = out_h + k_h - 1
      win = out_w + k_w - 1
      if hin and win and input_elems % (batch * hin * win) == 0:
        cin = input_elems // (batch * hin * win)
        add_config(batch, 1, cin, cout, out_h, out_w, k_h, k_w)
    if n == 7:
      batch, cout, out_h, out_w, cin, k_h, k_w = parts
      add_config(batch, 1, cin, cout, out_h, out_w, k_h, k_w)
      batch, groups, cout_pg, out_h, out_w, k_h, k_w = parts
      cout = cout_pg * groups
      hin = out_h + k_h - 1
      win = out_w + k_w - 1
      if hin and win and input_elems % (batch * hin * win) == 0:
        cin = input_elems // (batch * hin * win)
        add_config(batch, groups, cin, cout, out_h, out_w, k_h, k_w)

    if not configs: return None
    return max(configs, key=lambda cfg: (cfg.cin, -cfg.groups, cfg.cout))
  def _score_conv_config(self, config:ConvConfig, parts_set:set[int]) -> tuple[int, int, int]:
    dims = [config.batch, config.groups, config.cout_per_group, config.cin_per_group, *config.out_shape, *config.kernel_shape]
    score = sum(1 for d in dims if d > 1 and d in parts_set)
    if config.ndim == 2:
      score += 1
    out_prod = math.prod(config.out_shape)
    if out_prod in parts_set:
      score += 4
    kernel_prod = math.prod(config.kernel_shape)
    if kernel_prod in parts_set:
      score += 2
    return (score, -config.ndim, len(config.out_shape))

  def _rank_parts(self, config:ConvConfig, parts:list[int]) -> int:
    if not parts:
      return 0
    rank = 0
    parts_set = set(parts)
    rank += math.prod(config.out_shape) * 100
    rank += math.prod(config.kernel_shape) * 10
    def add(cond:bool, value:int) -> None:
      nonlocal rank
      if cond: rank += value
    if config.ndim == 1:
      if len(parts) >= 1:
        add(parts[0] == config.batch, 900)
        add(parts[0] == config.cout, 600)
      if len(parts) >= 2:
        add(parts[1] == config.cout, 700)
        add(parts[1] == config.out_shape[0], 500)
      if len(parts) >= 3:
        add(parts[2] == config.out_shape[0], 550)
        add(parts[2] in (config.cin, config.kernel_shape[0], config.cin_per_group), 400)
      if len(parts) >= 4: add(parts[3] == config.kernel_shape[0], 350)
    else:
      if len(parts) >= 1:
        add(parts[0] == config.batch, 950)
        add(parts[0] == config.cout, 700)
      if len(parts) >= 2:
        add(parts[1] == config.cout, 750)
        add(parts[1] == config.out_shape[0], 650)
      if len(parts) >= 3:
        add(parts[2] == config.out_shape[0], 620)
        add(parts[2] == config.out_shape[1], 600)
      if len(parts) >= 4: add(parts[3] in (config.cin, config.cin_per_group, config.groups), 550)
      if len(parts) >= 5: add(parts[4] == config.kernel_shape[0], 500)
      if len(parts) >= 6: add(parts[5] == config.kernel_shape[1], 450)
      if config.out_shape[0] <= config.out_shape[1]: rank += 500
      else: rank -= 250
      if config.in_shape_tail[0] <= config.in_shape_tail[1]: rank += 400
      else: rank -= 200
    attr_matches = {config.batch, config.cout, config.cin, config.groups,
                    *config.out_shape, *config.kernel_shape,
                    config.cout_per_group, config.cin_per_group}
    rank += sum(25 for p in parts if p in attr_matches)
    if config.groups > 1 and config.groups in parts_set:
      rank += 800
    if config.cin_per_group != config.cin and config.cin_per_group in parts_set:
      rank += 450
    if config.cout_per_group != config.cout and config.cout_per_group in parts_set:
      rank += 450
    return rank

  def _enumerate_conv1d(self, parts_set:set[int], values:set[int], weights:int, output:int, input_:int) -> list[ConvConfig]:
    configs:list[ConvConfig] = []
    for cout in _divisors(weights):
      if cout <= 0: continue
      for kW in _divisors(weights // cout):
        if kW <= 0: continue
        base = weights // (cout * kW)  # equals cin/groups
        if base <= 0: continue
        for groups in _divisors(cout):
          if groups <= 0: continue
          cin = base * groups
          if cin <= 0: continue
          if weights != cout * (cin // groups) * kW: continue
          for batch in _divisors(output // cout):
            if batch <= 0: continue
            if output % (batch * cout) != 0: continue
            Lout = output // (batch * cout)
            if Lout <= 0: continue
            Lin = Lout + kW - 1
            if Lin <= 0: continue
            if input_ != batch * cin * Lin: continue
            configs.append(ConvConfig(
              ndim=1,
              batch=batch,
              groups=groups,
              cin=cin,
              cout=cout,
              cin_per_group=cin // groups,
              cout_per_group=cout // groups,
              out_shape=(Lout,),
              kernel_shape=(kW,),
              in_shape_tail=(Lin,),
            ))
    return configs

  def _enumerate_conv2d(self, parts_set:set[int], values:set[int], weights:int, output:int, input_:int) -> list[ConvConfig]:
    configs:list[ConvConfig] = []
    for G in values:
      if G <= 0: continue
      for kH in _divisors(weights):
        if kH <= 0: continue
        if weights % kH != 0: continue
        remaining_for_kw = weights // kH
        for kW in _divisors(remaining_for_kw):
          if kW <= 0: continue
          kernel_prod = kH * kW
          if kernel_prod == 0 or weights % kernel_prod != 0: continue
          leftover = weights // kernel_prod
          for Cin_per_group in _divisors(leftover):
            if Cin_per_group <= 0: continue
            Cout = leftover // Cin_per_group
            if Cout <= 0 or Cout % G != 0: continue
            Cin = Cin_per_group * G
            Cout_per_group = Cout // G
            if Cout_per_group <= 0: continue
            if G > 1 and Cout_per_group not in parts_set: continue
            if Cin_per_group not in parts_set and Cin_per_group != 1: continue
            for N in _divisors(output):
              if N <= 0: continue
              denom = N * Cout
              if denom == 0 or output % denom != 0: continue
              rem = output // denom
              if rem <= 0: continue
              for Hout in _divisors(rem):
                if Hout <= 0: continue
                Wout = rem // Hout
                if Wout <= 0: continue
                Hin = Hout + kH - 1
                Win = Wout + kW - 1
                if Hin <= 0 or Win <= 0: continue
                if input_ != N * Cin * Hin * Win: continue
                configs.append(ConvConfig(
                  ndim=2,
                  batch=N,
                  groups=G,
                  cin=Cin,
                  cout=Cout,
                  cin_per_group=Cin_per_group,
                  cout_per_group=Cout_per_group,
                  out_shape=(Hout, Wout),
                  kernel_shape=(kH, kW),
                  in_shape_tail=(Hin, Win),
                ))
    return configs

  def _infer_conv_config(self, parts:list[int], buf_sizes:list[int], kernel_name:str) -> ConvConfig|None:
    if len(buf_sizes) < 3: return None
    if any(sz % 2 != 0 for sz in buf_sizes[:3]): return None
    weights = buf_sizes[2] // 2
    output = buf_sizes[0] // 2
    input_ = buf_sizes[1] // 2
    if weights <= 0 or output <= 0 or input_ <= 0: return None
    parts_set = set(parts)
    values = parts_set | {1}
    scored:list[tuple[tuple[int, int, int, int], ConvConfig]] = []
    conv1d_configs = self._enumerate_conv1d(parts_set, values, weights, output, input_)
    if conv1d_configs and getenv("DEBUG") >= 3:
      print("conv1d candidates", kernel_name, [(cfg.batch, cfg.groups, cfg.cin, cfg.cout, cfg.out_shape, cfg.kernel_shape) for cfg in conv1d_configs])
    for cfg in conv1d_configs:
      score0, score1, score2 = self._score_conv_config(cfg, parts_set)
      rank = self._rank_parts(cfg, parts)
      primary = score0 * 10000 + rank
      priority = (
        primary,
        score0,
        int(cfg.ndim == 2),
        int(cfg.cout in parts_set),
        int(cfg.cin in parts_set),
        int(cfg.batch in parts_set),
        int(cfg.groups in parts_set and cfg.groups > 1),
        sum(1 for d in cfg.out_shape if d in parts_set),
        sum(1 for d in cfg.kernel_shape if d in parts_set),
        score1,
        score2,
      )
      if getenv("DEBUG") >= 4:
        print("conv1d score", kernel_name, cfg, (score0, score1, score2, rank), "priority", priority)
      scored.append((priority, cfg))
    conv2d_configs = self._enumerate_conv2d(parts_set, values, weights, output, input_)
    if conv2d_configs and getenv("DEBUG") >= 3:
      print("conv2d candidates", kernel_name, [(cfg.batch, cfg.groups, cfg.cin, cfg.cout, cfg.out_shape, cfg.kernel_shape) for cfg in conv2d_configs])
    for cfg in conv2d_configs:
      score0, score1, score2 = self._score_conv_config(cfg, parts_set)
      rank = self._rank_parts(cfg, parts)
      primary = score0 * 10000 + rank
      priority = (
        primary,
        score0,
        int(cfg.ndim == 2),
        int(cfg.cout in parts_set),
        int(cfg.cin in parts_set),
        int(cfg.batch in parts_set),
        int(cfg.groups in parts_set and cfg.groups > 1),
        sum(1 for d in cfg.out_shape if d in parts_set),
        sum(1 for d in cfg.kernel_shape if d in parts_set),
        score1,
        score2,
      )
      if getenv("DEBUG") >= 4:
        print("conv2d score", kernel_name, cfg, (score0, score1, score2, rank), "priority", priority)
      scored.append((priority, cfg))
    if scored:
      scored.sort()
      return scored[-1][1]
    config = self._parse_conv1d_from_name(parts, buf_sizes)
    if config is not None: return config
    config = self._parse_conv2d_from_name(parts, buf_sizes)
    if config is not None: return config
    return None

  def _execute_conv(self, bufs:tuple[Any, ...], config:ConvConfig, kernel_name:str) -> None:
    out_buf, in_buf, weight_buf = bufs[:3]
    dtype_size = np.dtype(np.float16).itemsize
    in_elems = config.batch * config.cin * math.prod(config.in_shape_tail)
    w_elems = config.cout * config.cin_per_group * math.prod(config.kernel_shape)
    out_elems = config.batch * config.cout * math.prod(config.out_shape)
    in_mv = self._buffer_as_bytes(in_buf, in_elems * dtype_size)
    w_mv = self._buffer_as_bytes(weight_buf, w_elems * dtype_size)
    inp = np.frombuffer(in_mv, dtype=np.float16, count=in_elems).reshape((config.batch, config.cin, *config.in_shape_tail))
    if config.ndim == 1:
      weight_shape = (config.cout, config.cin_per_group, config.kernel_shape[0])
    else:
      weight_shape = (config.cout, config.cin_per_group, *config.kernel_shape)
    weight = np.frombuffer(w_mv, dtype=np.float16, count=w_elems).reshape(weight_shape)
    result = np.empty((config.batch, config.cout, *config.out_shape), dtype=np.float16)
    tile = math.prod(config.out_shape)
    K = config.cin_per_group * math.prod(config.kernel_shape)
    for g in range(config.groups):
      cin_slice = slice(g * config.cin_per_group, (g + 1) * config.cin_per_group)
      cout_slice = slice(g * config.cout_per_group, (g + 1) * config.cout_per_group)
      inp_group = inp[:, cin_slice, ...]
      weight_group = weight[cout_slice]
      if config.ndim == 1:
        windows = np.lib.stride_tricks.sliding_window_view(inp_group, window_shape=config.kernel_shape[0], axis=-1)
        col = windows.transpose(0, 2, 1, 3).reshape(config.batch * config.out_shape[0], K)
      else:
        windows = np.lib.stride_tricks.sliding_window_view(inp_group, window_shape=config.kernel_shape, axis=(-2, -1))
        col = windows.transpose(0, 2, 3, 1, 4, 5).reshape(config.batch * config.out_shape[0] * config.out_shape[1], K)
      col_arr = np.ascontiguousarray(col, dtype=np.float16)
      weight_mat = weight_group.reshape(config.cout_per_group, K)
      w_arr = np.ascontiguousarray(weight_mat.T, dtype=np.float16)
      for b in range(config.batch):
        col_slice = np.ascontiguousarray(col_arr[b*tile:(b+1)*tile])
        out_temp = bytearray(tile * config.cout_per_group * dtype_size)
        self._run_matmul_conv((out_temp, col_slice, w_arr), tile, K, config.cout_per_group)
        out_matrix = np.frombuffer(out_temp, dtype=np.float16, count=tile * config.cout_per_group)
        if config.ndim == 1:
          out_matrix = out_matrix.reshape(config.out_shape[0], config.cout_per_group).T
          result[b, cout_slice, :] = out_matrix
        else:
          out_matrix = out_matrix.reshape(config.out_shape[0], config.out_shape[1], config.cout_per_group)
          out_matrix = np.moveaxis(out_matrix, -1, 0)
          result[b, cout_slice, ...] = out_matrix
    out_bytes = result.tobytes()
    if isinstance(out_buf, HCQBuffer):
      ctypes.memmove(out_buf.va_addr, out_bytes, len(out_bytes))
    else:
      mv = self._buffer_as_bytes(out_buf, len(out_bytes))
      mv[:] = out_bytes
    if getenv("DEBUG"):
      try:
        dims = (config.batch, config.cout, *config.out_shape)
        if math.prod(dims) <= 4096:
          ref = np.zeros(dims, dtype=np.float32)
          for b in range(config.batch):
            for g in range(config.groups):
              cin_slice = slice(g * config.cin_per_group, (g + 1) * config.cin_per_group)
              cout_slice = slice(g * config.cout_per_group, (g + 1) * config.cout_per_group)
              inp_group = inp[b, cin_slice, ...].astype(np.float32)
              w_group = weight[cout_slice].astype(np.float32)
              if config.ndim == 1:
                for co in range(config.cout_per_group):
                  for idx in range(config.out_shape[0]):
                    ref[b, cout_slice.start + co, idx] = np.sum(
                      inp_group[:, idx:idx+config.kernel_shape[0]] * w_group[co]
                    )
              else:
                for co in range(config.cout_per_group):
                  for y in range(config.out_shape[0]):
                    for x in range(config.out_shape[1]):
                      ref[b, cout_slice.start + co, y, x] = np.sum(
                        inp_group[:, y:y+config.kernel_shape[0], x:x+config.kernel_shape[1]] * w_group[co]
                      )
          ref = ref.astype(np.float16)
          max_diff = np.max(np.abs(ref - result))
          if max_diff > 1e-3:
            print("conv debug mismatch", kernel_name, max_diff)
            if getenv("DEBUG") >= 4:
              flat_idx = np.unravel_index(np.argmax(np.abs(ref - result)), ref.shape)
              print("conv debug detail", kernel_name, flat_idx, result[flat_idx], ref[flat_idx])
      except Exception as exc:
        print("conv debug ref failed", exc)

  def _maybe_run_conv(self, bufs:tuple[Any, ...]) -> bool:
    name = getattr(self, "name", "")
    if not name.startswith("r_"): return False
    try:
      parts = [int(p) for p in name.split('_')[1:]]
    except ValueError:
      return False
    if len(parts) >= 3:
      try:
        dtype_size = 2  # assume fp16 paths for Rockchip conv/matmul kernels
        M, K, N = parts[0], parts[1], parts[2]
        if len(bufs) >= 3:
          buf_sizes = [self._buffer_nbytes(b) for b in bufs[:3]]
          if buf_sizes == [M * N * dtype_size, M * K * dtype_size, K * N * dtype_size]:
            return False
      except Exception:
        pass
    if len(parts) < 3:
      return False
    if len(bufs) < 3:
      return False
    try:
      buf_sizes = [self._buffer_nbytes(b) for b in bufs[:3]]
      if getenv("DEBUG") >= 3:
        print("conv candidate", name, "buf_sizes", buf_sizes)
    except Exception:
      return False
    config = self._infer_conv_config(parts, buf_sizes, name)
    if config is None:
      return False
    self._execute_conv(bufs, config, name)
    if getenv("DEBUG") >= 3:
      print("conv handled", name, config)
    return True

  def _build_matmul_queue(self, Mpad:int, Kpad:int, Npad:int,
                          input_dma:int, weight_dma:int, output_dma:int) -> list[int]:
    if (Mpad % MATMUL_CHANNEL_ALIGN != 0 or
        Kpad % MATMUL_CHANNEL_ALIGN != 0 or
        Npad % MATMUL_CHANNEL_ALIGN != 0):
      raise RuntimeError("unsupported matmul configuration for Rockchip template")

    tasks_arr = (ctypes.c_uint64 * 112)()
    params = MatmulParams()
    params.m = Mpad
    params.k = Kpad
    params.n = Npad
    params.input_dma = input_dma & 0xffffffff
    params.weights_dma = weight_dma & 0xffffffff
    params.output_dma = output_dma & 0xffffffff
    params.tasks = ctypes.cast(tasks_arr, ctypes.POINTER(ctypes.c_uint64))
    params.fp32tofp16 = 0
    ret = _gen_matmul_fp16(params)
    if ret != 0:
      raise RuntimeError(f"gen_matmul_fp16 returned {ret}")
    return [tasks_arr[i] for i in range(112)]

  def _submit_queue(self, q_vals:list[int], op_idx:int, enable_mask:int):
    tasks = ctypes.cast(self.device.task_buf.va_addr, ctypes.POINTER(rk.struct_rknpu_task * 128)).contents
    regcmd = ctypes.cast(self.device.cmd_buf.va_addr, ctypes.POINTER(ctypes.c_uint64 * 128)).contents
    for idx, val in enumerate(q_vals):
      regcmd[idx] = val
    tasks[0].flags = 0
    tasks[0].op_idx = op_idx
    tasks[0].enable_mask = enable_mask
    tasks[0].int_mask = 0x300
    tasks[0].int_clear = 0x1ffff
    tasks[0].int_status = 0
    tasks[0].regcfg_amount = max(len(q_vals) - (rk.RKNPU_PC_DATA_EXTRA_AMOUNT + 4), 0)
    tasks[0].regcfg_offset = 0
    tasks[0].regcmd_addr = self.device.cmd_buf.meta.dma_addr
    submit_res = rk.struct_rknpu_submit(
      flags=rk.RKNPU_JOB_PC | rk.RKNPU_JOB_BLOCK | rk.RKNPU_JOB_PINGPONG,
      timeout=20000,
      task_start=0,
      task_number=1,
      task_counter=0,
      priority=0,
      task_obj_addr=self.device.task_buf.meta.obj_addr,
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
    rk.DRM_IOCTL_RKNPU_SUBMIT(self.device.fd_ctl, __payload=submit_res)

  def _buffer_nbytes(self, buf) -> int:
    if isinstance(buf, HCQBuffer): return buf.size
    mv = memoryview(buf)
    return mv.nbytes

  def _buffer_as_bytes(self, buf, nbytes:int) -> memoryview:
    if isinstance(buf, HCQBuffer):
      if nbytes > buf.size:
        raise ValueError(f"buffer too small ({buf.size}) for requested bytes {nbytes}")
      return to_mv(ctypes.cast(int, buf.va_addr), nbytes)
    mv = memoryview(buf)
    if mv.format != 'B': mv = mv.cast('B')
    if mv.nbytes < nbytes: raise ValueError(f"buffer too small ({mv.nbytes}) for requested bytes {nbytes}")
    return mv[:nbytes]

  def _trace_define_global(self, idx:int) -> int|None:
    seen:set[int] = set()
    while idx not in seen:
      seen.add(idx)
      uop, _, srcs, _ = self.uops[idx]
      if uop is Ops.DEFINE_GLOBAL: return idx
      if uop in (Ops.LOAD, Ops.INDEX, Ops.CAST, Ops.BITCAST, Ops.GEP):
        if not srcs: return None
        idx = srcs[0]
        continue
      return None
    return None

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    st = time.perf_counter()
    if self._maybe_run_matmul(bufs):
      return time.perf_counter() - st
    if getenv("DEBUG"):
      print("rockchip execute fallback", getattr(self, "name", ""))
    warp = list(itertools.product(*[range(x) for x in local_size[::-1]]))
    warp_size = len(warp)
    define_indices = [idx for idx, uop in enumerate(self.uops) if uop[0] is Ops.DEFINE_GLOBAL]
    global_bufs = {idx: bufs[pos] for pos, idx in enumerate(define_indices)}
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
      
        elif uop in GroupOp.ALU:
          assert all_same([dtype] + dtp) or uop in {Ops.CMPNE, Ops.CMPLT, Ops.WHERE}, f"dtype mismatch on {uop}"
          handled = False

          lengths = [len(arr) for arr in inp]
          max_len = max(lengths) if lengths else 1
          can_broadcast = all(l in (max_len, 1) for l in lengths)
          if can_broadcast and max_len != 0 and not all_same(lengths):
            broadcasted:list[list[Any]] = []
            for arr in inp:
              if len(arr) == max_len:
                broadcasted.append(arr)
              elif len(arr) == 1:
                broadcasted.append([arr[0]] * max_len)
              else:
                can_broadcast = False
                break
            if can_broadcast:
              inp = broadcasted
              lengths = [max_len for _ in inp]

          debug_level = getenv("DEBUG")
          enable_alu_hw = getenv("ROCKCHIP_ALU_HW", 0)
          if (enable_alu_hw
            and len(inp) == 2
            and (dtype in (dtypes.float, dtypes.float16))
            and (uop in RockchipRenderer.code_for_op.keys())
            and lengths and all_same(lengths)):

            if debug_level >= 3:
              print("rockchip alu exec", uop, dtype, lengths, type(inp[0][0]))
            elem_count = lengths[0]
            if elem_count != 0:
              element_size = dtype.itemsize if hasattr(dtype, "itemsize") else dtype.base.itemsize
              self.device.add_buffer(elem_count * element_size)

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
                dst = np.frombuffer((bytearray(self.output_buf.size * dtypes.float16.itemsize)), dtype=np.float16)

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

              if getenv("ROCKTRACE"):
                print("rock_add_seq_default", [hex(int(v)) for v in self.q])
              self.submit()
              ctypes.memmove(dst.ctypes.data, self.output_buf.va_addr, self.output_buf.size * dtype.itemsize)
              ul[i] = dst.tolist()
              if debug_level >= 3:
                print("rockchip alu hardware", uop, dtype, elem_count)
              handled = True
            else:
              handled = False

          if not handled:
            if not lengths or not can_broadcast:
              if debug_level >= 3:
                print("rockchip alu bypass", uop, lengths)
            if can_broadcast and lengths and not all_same(lengths):
              max_len = max(lengths)
              inp = [[arr[0]] * max_len if len(arr) == 1 else arr for arr in inp]
              lengths = [max_len for _ in inp]
            if uop not in (Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE, Ops.XOR, Ops.AND, Ops.OR, Ops.TRUNC, Ops.ADD, Ops.MUL, Ops.IDIV, Ops.WHERE):
              print('ALLOWED FALLBACK TO CPU', uop, dtype)
            if lengths and lengths[0] and all_same(lengths):
              ul[i] = [exec_alu(uop, dtype, p) for p in zip(*inp)]
              handled = True
            if not handled and i not in ul:
              ul[i] = [exec_alu(uop, dtype, p) for p in zip(*inp)]

        assert i in ul, (uop, dtype, idp, arg)
        i += 1
    return time.perf_counter() - st

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
