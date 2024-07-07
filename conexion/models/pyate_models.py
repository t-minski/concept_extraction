from pyate import combo_basic, basic, cvalues
from conexion.models.base_model import BaseModel
from typing import List, Tuple

class PyateBasicsEntities(BaseModel):
    
    def __init__(self):
        pass

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[str]]:        
        # Extract keywords using Pyate entities
        entities = []
        for abstract in abstracts:
            pyate_keywords = basic(abstract)
            keywords_with_scores = [(keyword, score) for keyword, score in pyate_keywords.items()]
            entities.append(keywords_with_scores)
        
        return entities
    
class PyateComboBasicEntities(BaseModel):
    
    def __init__(self):
        pass

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[str]]:        
        # Extract keywords using Pyate entities
        entities = []
        for abstract in abstracts:
            pyate_keywords = combo_basic(abstract)
            keywords_with_scores = [(keyword, score) for keyword, score in pyate_keywords.items()]
            entities.append(keywords_with_scores)
        
        return entities
    
class PyateCvaluesEntities(BaseModel):
    
    def __init__(self):
        pass

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[str]]:        
        # Extract keywords using Pyate entities
        entities = []
        for abstract in abstracts:
            pyate_keywords = cvalues(abstract)
            keywords_with_scores = [(keyword, score) for keyword, score in pyate_keywords.items()]
            entities.append(keywords_with_scores)
        
        return entities