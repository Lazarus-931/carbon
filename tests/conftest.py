import os

import pytest

os.environ.setdefault("CARBON_SEARCH", "0")


def pytest_addoption(parser):
    parser.addoption(
        "--sweep",
        action="store_true",
        help="also run the long case tables marked `sweep`",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "sweep: a long case table, run with --sweep")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--sweep"):
        return
    skip = pytest.mark.skip(reason="sweep case: run with --sweep")
    for item in items:
        if "sweep" in item.keywords:
            item.add_marker(skip)
