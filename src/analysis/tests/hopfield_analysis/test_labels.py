from hopfield_analysis import labels


def test_translates_known_names_and_keeps_unknown_ones():
    assert labels.mode("synchronous") == "synchroniczny"
    assert labels.mode("asynchronous") == "asynchroniczny"
    assert labels.rule("hebb") == "Hebb"
    assert labels.rule("oja (subspace)") == "Oja (podprzestrzeń)"
    assert labels.rule("oja (single-neuron)") == "Oja (jeden neuron)"
    assert labels.final_class("spurious") == "stan pozorny"
    assert labels.final_class("unconverged") == "brak zbieżności"
    assert labels.rule("storkey") == "storkey"


def test_formats_numbers_with_a_decimal_comma_and_a_minus_sign():
    assert labels.number(-98.984127) == "−98,98"
    assert labels.number(0.5, 3) == "0,500"
    assert labels.number(3, 0) == "3"
