from yake import KeywordExtractor
from models.base_model import BaseModel
from typing import List, Tuple

class YakeEntities(BaseModel):
    
    def __init__(self):
        self.kw_extractor = KeywordExtractor(lan="en", n=2,
                                    dedupLim=0.81, dedupFunc='seqm',
                                    windowsSize=1, top=37)

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[str]]:        
        # Extract keywords using YAKE entities
        entities = []
        for abstract in abstracts:
            yake_keywords = self.kw_extractor.extract_keywords(abstract)
            keywords_with_scores = [(keyword[0], keyword[1]) for keyword in yake_keywords]
            entities.append(keywords_with_scores)
        
        return entities