from conexion.models.llm_confidence_models import split_keywords


ground_truth = ['quantum', 'quantum lattice-gas algorithms (QLGA)', 'nuclear magnetic resonance (NMR)', 'ensemble', 'pulse techniques', 'implementation errors']


def test_extraction__based_in_comma():
    text = "quantum, quantum lattice-gas algorithms (QLGA), nuclear magnetic resonance (NMR), ensemble, pulse techniques, implementation errors"
    assert(split_keywords(text) == ground_truth)
    text += ", , , , "
    assert(split_keywords(text) == ground_truth)

def test_extraction__based_in_semicolon():
    text = "quantum; quantum lattice-gas algorithms (QLGA); nuclear magnetic resonance (NMR); ensemble; pulse techniques; implementation errors"
    assert(split_keywords(text) == ground_truth)
    text += "; ; ; ; ;"
    assert(split_keywords(text) == ground_truth)

def test_extraction_based_on_stars():
    text = """
* quantum
* quantum lattice-gas algorithms (QLGA)
* nuclear magnetic resonance (NMR)
* ensemble
* pulse techniques
* implementation errors
"""
    assert(split_keywords(text) == ground_truth)

def test_extraction_based_on_newline():
    text = """
    quantum
    quantum lattice-gas algorithms (QLGA)
nuclear magnetic resonance (NMR)
             ensemble
      pulse techniques
               implementation errors
"""
    assert(split_keywords(text) == ground_truth)

    