"""Version metadata consistency tests."""

from importlib.metadata import PackageNotFoundError, version

import japan_fiscal_simulator as jpfs


def test_dunder_version_matches_distribution_metadata() -> None:
    """__version__ と配布メタデータの整合性を保証する。"""
    try:
        dist_version = version("jpfs")
    except PackageNotFoundError:
        assert jpfs.__version__ == "0+unknown"
        return

    assert jpfs.__version__ == dist_version
