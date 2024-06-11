from pyate import combo_basic, basic, cvalues
from models.base_model import BaseModel
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
            keywords = basic(abstract).sort_values(ascending=False).index.str.split().str[0].tolist()
            entities.append([(ent, 1.0) for ent in keywords])
        
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
            keywords = combo_basic(abstract).sort_values(ascending=False).index.str.split().str[0].tolist()
            entities.append([(ent, 1.0) for ent in keywords])
        
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
            keywords = cvalues(abstract).sort_values(ascending=False).index.str.split().str[0].tolist()
            entities.append([(ent, 1.0) for ent in keywords])
        
        return entities