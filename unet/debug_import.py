import sys
import os
print(f"CWD: {os.getcwd()}")
print(f"sys.path: {sys.path}")
try:
    import train_lib
    print(f"train_lib location: {train_lib.__file__}")
    from train_lib import cli
    print(f"cli location: {cli.__file__}")
    import inspect
    print(f"build_arg_parser source:\n{inspect.getsource(cli.build_arg_parser)}")
except ImportError as e:
    print(f"ImportError: {e}")
