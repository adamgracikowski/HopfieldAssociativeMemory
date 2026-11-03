from importlib.metadata import version

import hopfield_analysis


def test_version_matches_distribution_metadata():
    assert hopfield_analysis.__version__ == version("hopfield-analysis")
