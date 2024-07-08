from conexion.models import get_models
from conexion.models.spacy_models import SpacyEntities

def test_parsing():
    models = get_models(["SpacyEntities"])
    assert type(models[0]) == SpacyEntities

def test_parsing_two():
    models = get_models(["class=SpacyEntities"])
    assert type(models[0]) == SpacyEntities

def test_parsing_three():
    models = get_models(["class=Llama2_7b_Entities"])
