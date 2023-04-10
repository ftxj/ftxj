import setuptools

setuptools.setup(
    name="ftxj",
    version="0.0.1",
    author="Jie Xin",
    author_email="cs.xinjie@gmail.com",
    description="A profiling tool for python & cuda ",
    packages=setuptools.find_packages("python"),
    package_dir={"": "python"},
    ext_modules=[
        setuptools.Extension(
            "ftxj.profiler",
            sources=[
                "src/event.cpp",
                "src/interface.cpp",
                "src/python_tracer.cpp",
                "src/queue.cpp",
                "src/util.cpp"
            ],
        )
    ],
)
