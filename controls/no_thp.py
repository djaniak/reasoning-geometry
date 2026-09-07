"""Run a Python module with Linux transparent huge pages disabled.

Usage: .venv/bin/python -m controls.no_thp MODULE [ARG ...]
Ported from the existing argon helper; changes only this process tree.
"""

import ctypes
import runpy
import sys


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m controls.no_thp MODULE [ARG ...]")
    if sys.platform != "linux":
        raise SystemExit("controls.no_thp requires Linux prctl")
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(41, 1, 0, 0, 0) != 0:  # PR_SET_THP_DISABLE
        raise SystemExit(f"PR_SET_THP_DISABLE failed: errno {ctypes.get_errno()}")
    if libc.prctl(42, 0, 0, 0, 0) != 1:  # PR_GET_THP_DISABLE
        raise SystemExit("PR_SET_THP_DISABLE did not take")
    module = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
