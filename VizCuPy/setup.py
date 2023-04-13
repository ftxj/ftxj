import setuptools

from torch.utils.cpp_extension import (CUDA_HOME, BuildExtension, CppExtension,
                                       CUDAExtension)

setuptools.setup(
    name="ftxj",
    version="0.0.1",
    author="Jie Xin",
    author_email="cs.xinjie@gmail.com",
    description="A profiling tool for python & cuda ",
    packages=setuptools.find_packages("python"),
    package_dir={"": "python"},
    ext_modules=[
        CUDAExtension(
            "ftxj.profiler",
            include_dirs = ['/usr/local/cuda/include'],
            libraries = ['cuda', 'cupti'],
            library_dirs = ['/usr/local/cuda/lib64'],
            sources=[
                "src/event.cpp",
                "src/interface.cpp",
                "src/python_tracer.cpp",
                # "src/cuda_tracer.cpp",
                "src/queue.cpp",
                "src/util.cpp",
                "src/timeline_schedule.cpp"
            ],
        )
    ],
)
