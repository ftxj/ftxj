import argparse

parser = argparse.ArgumentParser(
    prog="JitPy",
    description="JitPy Interpreter.",
)

parser.add_argument(
    'prog',
    help="The program to run.",
)