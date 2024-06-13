import pke
from models.base_model import BaseModel
from typing import List, Tuple

class TopicalPageRankEntities(BaseModel):

    def __init__(self, method: str):
        self.method = 'TopicalPageRank'

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[Tuple[str, float]]]:
        entities = []
        for abstract in abstracts:
            extractor = pke.unsupervised.__dict__[self.method]()
            extractor.load_document(input=abstract, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keywords = extractor.get_n_best(n=1000000, stemming=False)  # Set a very high number to get all keywords
            entities.append(keywords)
        
        return entities