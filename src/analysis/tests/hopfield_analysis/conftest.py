from pathlib import Path

import pytest

from hopfield_analysis.recall import Recall

FIXTURES = Path(__file__).parents[1] / "fixtures"


@pytest.fixture
def recall() -> Recall:
    """The recall run of ``tests/fixtures/recall.toml`` written by ``compute``:
    ``animals-14x9`` (9 rows of 14 pixels), 10 frames, target pattern 4."""
    return Recall.load(FIXTURES / "recall")
