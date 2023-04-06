import setuptools

setuptools.setup(
    name="ftxj",
    version="0.0.1",
    author="Jie Xin",
    author_email="cs.xinjie@gmail.com",
    description="A profiling tool for python & cuda ",
    ext_modules=[
        setuptools.Extension(
            "ftxj.profiler",
            sources=[
                "src/profiler_base.cpp"
            ],
        )
    ],
)
