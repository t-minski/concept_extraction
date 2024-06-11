from pyate import combo_basic, basic, cvalues
from base_model import BaseModel

class PyateBasicsEntities(BaseModel):
    
    def __init__(self):
        pass

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[str]]:        
        # Extract keywords using Pyate entities
        entities = []
        for abstract in abstracts:
            pyate_basic_keywords = basic(abstract).sort_values(ascending=False).index.str.split().str[0].tolist()
            entities.append(pyate_basic_keywords)
        
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
            pyate_combo_basic_keywords = combo_basic(abstract).sort_values(ascending=False).index.str.split().str[0].tolist()
            entities.append(pyate_combo_basic_keywords)
        
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
            pyate_cvalues_keywords = cvalues(abstract).sort_values(ascending=False).index.str.split().str[0].tolist()
            entities.append(pyate_cvalues_keywords)
        
        return entities