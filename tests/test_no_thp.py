"""Check actual process flag inheritance without running a research analysis."""
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="Linux prctl only")
def test_module_arguments_and_thp_flag_survive_exec(tmp_path):
    module = tmp_path / "check_flag.py"
    module.write_text(
        "import ctypes, subprocess, sys\n"
        "assert sys.argv[1:] == ['sentinel']\n"
        "assert ctypes.CDLL('libc.so.6').prctl(42,0,0,0,0) == 1\n"
        "subprocess.run([sys.executable, '-c', "
        "\"import ctypes; assert ctypes.CDLL('libc.so.6').prctl(42,0,0,0,0)==1\"], check=True)\n"
    )
    import os
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "controls.no_thp", "check_flag", "sentinel"],
        cwd=root, env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
