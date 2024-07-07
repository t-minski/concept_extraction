from rake_nltk import Rake
from conexion.models.base_model import BaseModel
from typing import List, Tuple

class RakeEntities(BaseModel):
    
    def __init__(self):
        self.rake_nltk_var = Rake()

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[str]]:        
        # Extract keywords using Rake entities
        entities = []
        for abstract in abstracts:
            self.rake_nltk_var.extract_keywords_from_text(abstract)
            keywords = self.rake_nltk_var.get_ranked_phrases_with_scores()
            keywords_with_scores = [(keyword[1], keyword[0]) for keyword in keywords]
            entities.append(keywords_with_scores)
        
        return entities