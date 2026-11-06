from pathlib import Path

import pytest

from hopfield_analysis.recall import Recall

FIXTURES = Path(__file__).parents[1] / "fixtures"


@pytest.fixture
def recall() -> Recall:
    """The recall run of ``tests/fixtures/recall.toml`` written by ``compute``."""
    return Recall.load(FIXTURES / "recall")
