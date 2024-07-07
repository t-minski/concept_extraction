from summa import keywords as summa_keywords
from conexion.models.base_model import BaseModel
from typing import List, Tuple

class SummaEntities(BaseModel):
    
    def __init__(self):
        pass

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[str]]:        
        # Extract keywords using Summa entities
        entities = []
        for abstract in abstracts:
            keywords_with_scores = [(keyword[0], keyword[1]) for keyword in summa_keywords.keywords(abstract, scores=True)]
            entities.append(keywords_with_scores)
        
        return entities