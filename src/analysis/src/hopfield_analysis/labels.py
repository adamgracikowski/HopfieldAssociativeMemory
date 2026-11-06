"""Polish texts of the figures.

The compute layer writes names in English (``synchronous``, ``hebb``,
``spurious``); figures in the report show them in Polish. A name without a
translation is shown as it is.
"""

NOISE = "poziom szumu (odsetek odwróconych pikseli)"
RECOVERED = "odsetek odzyskanych"
FRACTION = "odsetek przebiegów"
FRAME = "klatka"
CHANGES = "zmiany"
ENERGY = "energia $E$"
OVERLAP = "overlap $m^\\mu$"
PATTERN = "wzorzec"
TARGET = "cel"
NEAREST = "najbliższy"
OTHER_PATTERNS = "pozostałe wzorce"
NEURON = "neuron"
WEIGHT = "waga $w_{ij}$"

MODES = {
    "synchronous": "synchroniczny",
    "asynchronous": "asynchroniczny",
}

RULES = {
    "hebb": "Hebb",
    "oja (subspace)": "Oja (podprzestrzeń)",
    "oja (single-neuron)": "Oja (jeden neuron)",
}

CLASSES = {
    "pattern": "wzorzec",
    "negation": "negacja",
    "spurious": "stan pozorny",
    "cycle": "cykl",
    "unconverged": "brak zbieżności",
}


def mode(name: str) -> str:
    """Polish name of an update mode, e.g. ``synchroniczny``."""
    return MODES.get(name, name)


def rule(name: str) -> str:
    """Polish name of a learning rule, e.g. ``Oja (podprzestrzeń)``."""
    return RULES.get(name, name)


def final_class(name: str) -> str:
    """Polish name of a final state class, e.g. ``stan pozorny``."""
    return CLASSES.get(name, name)


def number(value: float, digits: int = 2) -> str:
    """Formats ``value`` with ``digits`` decimals, a decimal comma and a minus sign.

    Args:
        value: The number.
        digits: Number of decimals.

    Returns:
        The text, e.g. ``-98,98`` for ``-98.984``.
    """
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "−")
