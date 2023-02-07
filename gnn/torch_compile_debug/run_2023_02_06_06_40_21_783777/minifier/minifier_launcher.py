
import os
from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import functools
import torch._dynamo
from torch._dynamo.debug_utils import run_fwd_maybe_bwd
from torch._dynamo.optimizations.backends import BACKENDS
from torch._dynamo.testing import rand_strided

import torch._dynamo.config
import torch._inductor.config
torch._dynamo.config.load_config(b'\x80\x04\x95\x83\x07\x00\x00\x00\x00\x00\x00}\x94(\x8c\x08__name__\x94\x8c\x14torch._dynamo.config\x94\x8c\x07__doc__\x94N\x8c\x0b__package__\x94\x8c\rtorch._dynamo\x94\x8c\n__loader__\x94\x8c\x1a_frozen_importlib_external\x94\x8c\x10SourceFileLoader\x94\x93\x94)\x81\x94}\x94(\x8c\x04name\x94h\x02\x8c\x04path\x94\x8c>/opt/conda/lib/python3.8/site-packages/torch/_dynamo/config.py\x94ub\x8c\x08__spec__\x94\x8c\x11_frozen_importlib\x94\x8c\nModuleSpec\x94\x93\x94)\x81\x94}\x94(h\x0ch\x02\x8c\x06loader\x94h\n\x8c\x06origin\x94h\x0e\x8c\x0cloader_state\x94N\x8c\x1asubmodule_search_locations\x94N\x8c\r_set_fileattr\x94\x88\x8c\x07_cached\x94\x8cV/opt/conda/lib/python3.8/site-packages/torch/_dynamo/__pycache__/config.cpython-38.pyc\x94\x8c\r_initializing\x94\x89ub\x8c\x08__file__\x94h\x0e\x8c\n__cached__\x94h\x1b\x8c\x07abspath\x94\x8c\tposixpath\x94h\x1f\x93\x94\x8c\x07dirname\x94h h"\x93\x94\x8c\x0eHAS_REFS_PRIMS\x94\x88\x8c\tlog_level\x94K\n\x8c\x0boutput_code\x94\x89\x8c\rlog_file_name\x94N\x8c\x07verbose\x94\x88\x8c\x11output_graph_code\x94\x89\x8c\x12verify_correctness\x94\x89\x8c\x12minimum_call_count\x94K\x01\x8c\x15dead_code_elimination\x94\x88\x8c\x10cache_size_limit\x94K@\x8c\x14specialize_int_float\x94\x88\x8c\x0edynamic_shapes\x94\x89\x8c\x10guard_nn_modules\x94\x89\x8c\x0cnormalize_ir\x94\x89\x8c\x1btraceable_tensor_subclasses\x94\x8f\x94\x8c\x0fsuppress_errors\x94\x89\x8c\x15replay_record_enabled\x94\x89\x8c rewrite_assert_with_torch_assert\x94\x88\x8c\x12print_graph_breaks\x94\x89\x8c\x07disable\x94\x89\x8c*allowed_functions_module_string_ignorelist\x94\x8f\x94(\x8c\x0btorch._refs\x94\x8c\rtorch.testing\x94\x8c\x0ctorch._prims\x94\x8c\rtorch._decomp\x94\x8c\x13torch.distributions\x94\x90\x8c\x16capture_scalar_outputs\x94\x89\x8c\x19enforce_cond_guards_match\x94\x88\x8c\x0coptimize_ddp\x94\x88\x8c\x1araise_on_ctx_manager_usage\x94\x88\x8c\x1craise_on_unsafe_aot_autograd\x94\x89\x8c\rdynamo_import\x94\x8c\rtorch._dynamo\x94\x8c\x0finductor_import\x94\x8c\x0ftorch._inductor\x94\x8c\x18error_on_nested_fx_trace\x94\x88\x8c\x08base_dir\x94\x8c&/opt/conda/lib/python3.8/site-packages\x94\x8c\x0edebug_dir_root\x94\x8c\'/workspace/ftxj/gnn/torch_compile_debug\x94\x8c)DO_NOT_USE_legacy_non_fake_example_inputs\x94\x89\x8c\x15_AccessLimitingConfig\x94}\x94(\x8c\n__module__\x94h\x02\x8c\x0b__setattr__\x94h\x02\x8c!_AccessLimitingConfig.__setattr__\x94\x93\x94h\x03Nu\x8c\x15_allowed_config_names\x94\x8f\x94(h6h4h8h/h2hEh"h\x0fhC\x8c!skipfiles_inline_module_allowlist\x94h&h\x03h\x1dh-hNh1h(\x8c\x05torch\x94\x8c\x0c__builtins__\x94h,hIh*hAh\x1fh+hG\x8c\x0brepro_level\x94hLhDhJ\x8c\x12constant_functions\x94h$h\'h7h\x1e\x8c\x07logging\x94h\x04h@\x8c\x0eexternal_utils\x94\x8c\nModuleType\x94h.h\x06\x8c\x02os\x94h\x01h0hOh5h9h%hBh)\x8c\x0brepro_after\x94\x8c\x03sys\x94\x90\x8c\x1cget_config_serialization_fns\x94\x8c\x1atorch._dynamo.config_utils\x94hb\x93\x94u.')
torch._inductor.config.load_config(b'\x80\x04\x95\x0e\t\x00\x00\x00\x00\x00\x00}\x94(\x8c\x08__name__\x94\x8c\x16torch._inductor.config\x94\x8c\x07__doc__\x94N\x8c\x0b__package__\x94\x8c\x0ftorch._inductor\x94\x8c\n__loader__\x94\x8c\x1a_frozen_importlib_external\x94\x8c\x10SourceFileLoader\x94\x93\x94)\x81\x94}\x94(\x8c\x04name\x94h\x02\x8c\x04path\x94\x8c@/opt/conda/lib/python3.8/site-packages/torch/_inductor/config.py\x94ub\x8c\x08__spec__\x94\x8c\x11_frozen_importlib\x94\x8c\nModuleSpec\x94\x93\x94)\x81\x94}\x94(h\x0ch\x02\x8c\x06loader\x94h\n\x8c\x06origin\x94h\x0e\x8c\x0cloader_state\x94N\x8c\x1asubmodule_search_locations\x94N\x8c\r_set_fileattr\x94\x88\x8c\x07_cached\x94\x8cX/opt/conda/lib/python3.8/site-packages/torch/_inductor/__pycache__/config.cpython-38.pyc\x94\x8c\r_initializing\x94\x89ub\x8c\x08__file__\x94h\x0e\x8c\n__cached__\x94h\x1b\x8c\x05debug\x94\x88\x8c\x10disable_progress\x94\x88\x8c\x10verbose_progress\x94\x89\x8c\x0bcpp_wrapper\x94\x89\x8c\x03dce\x94\x89\x8c\x0edynamic_shapes\x94\x89\x8c\x14static_weight_shapes\x94\x88\x8c\x0csize_asserts\x94\x88\x8c\x10pick_loop_orders\x94\x88\x8c\x0finplace_buffers\x94\x88\x8c\x11benchmark_harness\x94\x88\x8c\x17realize_reads_threshold\x94K\x04\x8c\x17realize_bytes_threshold\x94M\xd0\x07\x8c\x1brealize_acc_reads_threshold\x94K\x08\x8c\x0ffallback_random\x94\x89\x8c\x12implicit_fallbacks\x94\x88\x8c\rprefuse_nodes\x94\x88\x8c\x0btune_layout\x94\x89\x8c\x11aggressive_fusion\x94\x89\x8c\x0fmax_fusion_size\x94K@\x8c\x1bunroll_reductions_threshold\x94K\x08\x8c\x0ecomment_origin\x94\x89\x8c\tis_fbcode\x94h\x02h5\x93\x94\x8c\x0fcompile_threads\x94K \x8c\x13kernel_name_max_ops\x94K\n\x8c\x0finductor_import\x94\x8c\x0ftorch._inductor\x94\x8c\rdynamo_import\x94\x8c\rtorch._dynamo\x94\x8c\rshape_padding\x94\x89\x8c\x0epermute_fusion\x94\x89\x8c\x1aprofiler_mark_wrapper_call\x94\x89\x8c\x03cpp\x94}\x94(\x8c\n__module__\x94h\x02\x8c\x07threads\x94J\xff\xff\xff\xff\x8c\x0fdynamic_threads\x94\x89\x8c\x07simdlen\x94N\x8c\x0emin_chunk_size\x94M\x00\x10\x8c\x03cxx\x94N\x8c\x03g++\x94\x86\x94\x8c\x15enable_kernel_profile\x94\x89h\x03Nu\x8c\x06triton\x94}\x94(hBh\x02\x8c\ncudagraphs\x94\x88\x8c\x10debug_sync_graph\x94\x89\x8c\x11debug_sync_kernel\x94\x89\x8c\x0bconvolution\x94\x8c\x04aten\x94\x8c\x02mm\x94hQ\x8c\x0edense_indexing\x94\x89\x8c\tmax_tiles\x94K\x02\x8c\x08autotune\x94\x88\x8c\x07use_bmm\x94\x89\x8c tiling_prevents_pointwise_fusion\x94\x88\x8c tiling_prevents_reduction_fusion\x94\x88\x8c\x14ordered_kernel_names\x94\x89\x8c\x18descriptive_kernel_names\x94\x88h\x03Nu\x8c\x05trace\x94}\x94(hBh\x02\x8c\x07enabled\x94\x89\x8c\tdebug_log\x94\x88\x8c\x08info_log\x94\x89\x8c\x08fx_graph\x94\x88\x8c\rir_pre_fusion\x94\x88\x8c\x0eir_post_fusion\x94\x88\x8c\x0boutput_code\x94\x88\x8c\rgraph_diagram\x94\x89\x8c\x0fcompile_profile\x94\x89\x8c\nupload_tar\x94Nh\x03Nu\x8c\x15InductorConfigContext\x94}\x94(hBh\x02\x8c\x0f__annotations__\x94}\x94(\x8c\rstatic_memory\x94\x8c\x08builtins\x94\x8c\x04bool\x94\x93\x94\x8c\x0bmatmul_tune\x94hl\x8c\x03str\x94\x93\x94\x8c\x0ematmul_padding\x94hn\x8c\x0ftriton_autotune\x94hn\x8c\ntriton_bmm\x94hn\x8c\ttriton_mm\x94hq\x8c\x12triton_convolution\x94hq\x8c\x17rematerialize_threshold\x94hl\x8c\x03int\x94\x93\x94\x8c\x1brematerialize_acc_threshold\x94hyu\x8c\x05_save\x94h\x02\x8c\x1bInductorConfigContext._save\x94\x93\x94\x8c\x06_apply\x94h\x02\x8c\x1cInductorConfigContext._apply\x94\x93\x94\x8c\x08__init__\x94h\x02\x8c\x1eInductorConfigContext.__init__\x94\x93\x94\x8c\t__enter__\x94h\x02\x8c\x1fInductorConfigContext.__enter__\x94\x93\x94\x8c\x08__exit__\x94h\x02\x8c\x1eInductorConfigContext.__exit__\x94\x93\x94h\x03Nu\x8c\x1cget_config_serialization_fns\x94\x8c\x1atorch._dynamo.config_utils\x94h\x8a\x93\x94u.')


# REPLACEABLE COMMENT FOR TESTING PURPOSES

args = [((2708, 1433), (1433, 1), torch.float32, 'cuda', False)]
args = [rand_strided(sh, st, dt, dev).requires_grad_(rg) for (sh, st, dt, dev, rg) in args]


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_lin_src_weight = torch.nn.Parameter(torch.randn([64, 1433], dtype=torch.float32)).cuda()
        self.self_att_src = torch.nn.Parameter(torch.randn([1, 8, 8], dtype=torch.float32)).cuda()
        self.self_att_dst = torch.nn.Parameter(torch.randn([1, 8, 8], dtype=torch.float32)).cuda()



    def forward(self, x : torch.Tensor):
        self_lin_src_weight = self.self_lin_src_weight
        linear = torch._C._nn.linear(x, self_lin_src_weight, None);  x = self_lin_src_weight = None
        view = linear.view(-1, 8, 8);  linear = None
        self_att_src = self.self_att_src
        mul = view * self_att_src;  self_att_src = None
        sum_1 = mul.sum(dim = -1);  mul = None
        self_att_dst = self.self_att_dst
        mul_1 = view * self_att_dst;  self_att_dst = None
        sum_2 = mul_1.sum(-1);  mul_1 = None
        return (view, sum_1, sum_2)


mod = Repro()

# Setup debug minifier compiler
torch._dynamo.debug_utils.MINIFIER_SPAWNED = True
compiler_fn = BACKENDS["dynamo_minifier_backend"]

dynamo_minifier_backend = functools.partial(
    compiler_fn,
    compiler_name="inductor",
)
opt_mod = torch._dynamo.optimize(dynamo_minifier_backend)(mod)

with torch.cuda.amp.autocast(enabled=False):
    opt_mod(*args)
