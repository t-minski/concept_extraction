import spacy
from conexion.models.base_model import BaseModel
from typing import List, Tuple

class SpacyEntities(BaseModel):
    
    def __init__(self, spacy_model:str="en_core_web_lg"):
        self.spacy_model = spacy_model
        self.nlp = spacy.load(spacy_model)

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[Tuple[str, float]]]:
        # Extract keywords using Spacy entities
        entities = []
        for abstract in abstracts:
            doc = self.nlp(abstract)
            entities.append([(ent.text, 1.0) for ent in doc.ents])
        
        return entities


class SpacyNounChunks(BaseModel):
    
    def __init__(self, spacy_model:str="en_core_web_lg"):
        self.spacy_model = spacy_model
        self.nlp = spacy.load(spacy_model)

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[Tuple[str, float]]]:
        # Extract keywords using Spacy entities
        entities = []
        for abstract in abstracts:
            doc = self.nlp(abstract)
            entities.append([(chunk.text, 1.0) for chunk in doc.noun_chunks])
        return entities