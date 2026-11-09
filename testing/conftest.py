"""Shared pytest configuration for the testing/ directory."""

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip `golden_exact` tests unless selected with -m golden_exact: their
    bit-for-bit comparisons only hold on the machine (torch / BLAS build)
    that recorded the fixture."""
    if "golden_exact" in (config.getoption("markexpr") or ""):
        return
    skip = pytest.mark.skip(reason="bit-for-bit golden check; run with -m golden_exact")
    for item in items:
        if "golden_exact" in item.keywords:
            item.add_marker(skip)
