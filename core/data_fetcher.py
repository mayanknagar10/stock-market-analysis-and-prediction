"""Compatibility imports; provider operations are centralized under core.data."""
from core.data.legacy_market import *
from core.data.legacy_market import __name__ as _adapter_name

def __getattr__(name):
    import importlib
    return getattr(importlib.import_module(_adapter_name),name)
