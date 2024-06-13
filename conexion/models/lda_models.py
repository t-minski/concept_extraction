from gensim.models import LdaModel
from gensim.corpora import Dictionary
from models.base_model import BaseModel
from typing import List, Tuple

class LDAEntities(BaseModel):
    
    def __init__(self):
        self.num_topics = 18  # Adjust the number of topics as needed
        self.dictionary = None
        self.lda_model = None

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[str]]:        
        # Extract keywords using LDA entities
        entities = []
        for abstract in abstracts:
            bow = self.dictionary.doc2bow(abstract.split())
            topics = self.lda_model.get_document_topics(bow)
            
            # Get the top keywords for the dominant topic
            dominant_topic = max(topics, key=lambda x: x[1])[0]
            keywords_with_scores = self.lda_model.show_topic(dominant_topic)
            
            entities.append(keywords_with_scores)
        
        return entities