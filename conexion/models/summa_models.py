from summa import keywords as summa_keywords
from base_model import BaseModel

class RakeEntities(BaseModel):
    
    def __init__(self):
        pass

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[str]]:        
        # Extract keywords using Summa entities
        entities = []
        for abstract in abstracts:
            summa_keywords_ = [keyword[0] for keyword in summa_keywords.keywords(abstract, scores=True)]
            entities.append(summa_keywords_)
        
        return entities