from ctypes import c_void_p, c_long
import torch
import math
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()

import triton
import triton.language as tl
from torch._inductor.triton_ops.autotune import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_jxin/y6/cy6kzfjfj3bv73h3nb6y5czrmnu72qajcyaojkaczlavbvcx5tsl.py
# Original ATen: aten.index_select, aten.mul, aten.new_zeros, aten.scatter_add

# aten.index_select => index
# aten.mul => mul
# aten.new_zeros => full
# aten.scatter_add => scatter_add
triton_fused_index_select_mul_new_zeros_scatter_add_0 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3727440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_jxin/4i/c4ivehv2ctifflwh2frz34u6hodwfzlg2umy7xg37sps6y4cmqhq.py
# Original ATen: aten.index_select, aten.mul, aten.scatter_add

# aten.index_select => index
# aten.mul => mul
# aten.scatter_add => scatter_add
triton_fused_index_select_mul_scatter_add_1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2147483648], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1837581712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (114848857 + x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask)
    tmp2 = tl.load(in_ptr0 + (x1), xmask)
    tmp3 = tl.load(in_ptr2 + (x0 + (16*tmp2)), xmask)
    tmp4 = tmp1 * tmp3
    tl.atomic_add(out_ptr0 + (x0 + (16*tmp0) + tl.zeros([XBLOCK], tl.int32)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_jxin/74/c74tkkadtzl7tjnzpfwuh4ued5b6e3dyswjamnxxtbnzhx4bl6yx.py
# Original ATen: aten.add

# aten.add => add
triton_fused_add_2 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3727440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + tl.zeros([XBLOCK], tl.int32)), tmp2, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((232965, 16), (16, 1), device='cuda', dtype=torch.float32)
        extern_kernels.mm(primals_5, as_strided(primals_1, (602, 16), (1, 602)), out=buf0)
        del primals_1
        buf1 = empty_strided((232965, 16), (16, 1), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((232965, 16), (16, 1), device='cuda', dtype=torch.float32)
        stream0 = get_cuda_stream(0)
        triton_fused_index_select_mul_new_zeros_scatter_add_0.run(buf1, buf2, 3727440, grid=grid(3727440), stream=stream0)
        triton_fused_index_select_mul_scatter_add_1.run(primals_3, primals_4, buf0, buf2, 1837581712, grid=grid(1837581712), stream=stream0)
        buf4 = buf0; del buf0  # reuse
        triton_fused_add_2.run(buf2, primals_2, buf4, 3727440, grid=grid(3727440), stream=stream0)
        del buf2
        del primals_2
        return (buf4, primals_4, primals_5, as_strided(primals_3, (114848857, ), (1, )), as_strided(primals_3, (114848857, 16), (1, 0), 114848857), buf1, )


def benchmark_compiled_module():
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 602), (602, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((2, 114848857), (114848857, 1), device='cuda:0', dtype=torch.int64)
    primals_4 = rand_strided((114848857, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((232965, 602), (602, 1), device='cuda:0', dtype=torch.float32)
    print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5]))


if __name__ == "__main__":
    import argparse
    from torch._inductor.utils import benchmark_all_kernels

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-kernels", "-k", action="store_true", help="Whether to benchmark each individual kernels")
    parser.add_argument("--benchmark-all-configs", "-c", action="store_true", help="Whether to benchmark each individual config for a kernel")
    args = parser.parse_args()

    if args.benchmark_kernels:
        benchmark_all_kernels('None', args.benchmark_all_configs)
    else:
        benchmark_compiled_module()