"""
DGL compatibility shim.

DGL 2.1.0's graphbolt sub-package requires a C++ library compiled per
PyTorch version. When the .so is missing (e.g. PyTorch 2.10+), importing
DGL fails even though we never use graphbolt.

Call ``patch()`` *before* any ``import dgl`` to stub out the module.
"""

import sys
import types


def patch():
    key = "dgl.graphbolt"
    if key not in sys.modules:
        stub = types.ModuleType(key)
        stub.__path__ = []
        sys.modules[key] = stub
