#!/usr/bin/env python
"""In this check, we run the slow orbital and CI convergence tests together.
Plots are saved in the current directory. Instead of running these tests in each CI build.
"""

from pathlib import Path
import sys

import pytest


if __name__ == "__main__":
    test_file = (Path(__file__).resolve().parents[2] / "tests" / "kLASSCF"
                 / "test_pbc_klasscf_fd_convergence_slow.py")
    raise SystemExit(pytest.main([
        str(test_file), "-W", "ignore::DeprecationWarning:pyparsing.*",
        *sys.argv[1:],
    ]))
