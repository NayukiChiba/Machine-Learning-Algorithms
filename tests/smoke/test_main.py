"""
CLI 烟雾测试
"""

from __future__ import annotations

import importlib.util

import main as entryMain


def testMainListCommand(capsys):
    """list 命令应可执行。"""
    assert entryMain.main(["list"]) == 0
    output = capsys.readouterr().out
    assert "classification.logistic_regression" in output


def testMainAnalyzeCommand():
    """analyze 命令应可执行。"""
    assert entryMain.main(["analyze", "classification.logistic_regression"]) == 0


def testMainRunCommand():
    """run 命令应可执行。"""
    assert entryMain.main(["run", "classification.logistic_regression"]) == 0


def testMainSuiteCommand():
    """suite 命令应可执行。"""
    if importlib.util.find_spec("hmmlearn") is None:
        assert entryMain.main(["suite", "probabilistic"]) == 0
    else:
        assert entryMain.main(["suite", "probabilistic"]) == 0
