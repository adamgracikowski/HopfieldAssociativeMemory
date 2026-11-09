"""Polish texts of the figures.

The compute layer writes names in English (``synchronous``, ``hebb``,
``spurious``); figures in the report show them in Polish. A name without a
translation is shown as it is.
"""

NOISE = "poziom szumu (odsetek odwróconych pikseli)"
NOISE_SHORT = "poziom szumu"
RANDOM_INPUT = "losowe"
RECOVERED = "odsetek odzyskanych"
FRACTION = "odsetek przebiegów"
FRAME = "klatka"
CHANGES = "zmiany"
ENERGY = "energia $E$"
OVERLAP = "nakładanie $m^\\mu$"
PATTERN = "wzorzec"
TARGET = "cel"
NEAREST = "najbliższy"
OTHER_PATTERNS = "pozostałe wzorce"
NEURON = "neuron"
WEIGHT = "waga $w_{ij}$"
STABLE_FRACTION = "odsetek stabilnych wzorców"
FIELD = "pole $\\xi_i^\\mu h_i$"
NEURON_COUNT = "liczba neuronów"
UNSTABLE_PIXELS = "liczba niestabilnych pikseli (zaznaczone na czerwono)"
OVERLAP_MATRIX = "nakładanie $m^{\\mu\\nu}$"
LOAD = "obciążenie $p/N$"
STABLE_FRACTION_WRAPPED = "odsetek stabilnych\nwzorców"
BIT_ERRORS_WRAPPED = "odsetek niestabilnych\nneuronów"
RANDOM_PATTERNS = "wzorce losowe"
THEORY = "teoria (wzorce losowe, Hebb)"
HOPFIELD_LOAD = "$p/N = 0{,}138$"
STABLE_LOAD = "$p/N = 1/(2 \\ln N)$"

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


def decimal(value: float) -> str:
    """Formats ``value`` with the digits it needs, a decimal comma and a minus sign.

    Args:
        value: The number, e.g. a noise level.

    Returns:
        The shortest text that keeps 6 significant digits, e.g. ``0,025`` for
        ``0.025`` and ``0,1`` for ``0.1``.
    """
    return f"{value:g}".replace(".", ",").replace("-", "\u2212")
