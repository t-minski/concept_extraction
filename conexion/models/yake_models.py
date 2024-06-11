from yake import KeywordExtractor
from base_model import BaseModel

class SpacyEntities(BaseModel):
    
    def __init__(self):
        self.kw_extractor = KeywordExtractor(lan="en", n=2,
                                    dedupLim=0.81, dedupFunc='seqm',
                                    windowsSize=1, top=37)

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[str]]:        
        # Extract keywords using Spacy entities
        entities = []
        for abstract in abstracts:
            yake_keywords = self.kw_extractor.extract_keywords(abstract)
            yake_keywords = [keyword[0] for keyword in yake_keywords]
            entities.append(yake_keywords)
        
        return entities