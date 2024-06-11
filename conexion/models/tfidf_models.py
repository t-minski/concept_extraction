from sklearn.feature_extraction.text import TfidfVectorizer
from models.base_model import BaseModel
from typing import List, Tuple

class TfIdfEntities(BaseModel):
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
                                    stop_words='english',
                                    ngram_range=(1, 2),
                                    #max_df=max_df,
                                    min_df=0.49,
                                    #max_features=max_features,
                                    norm=None,
                                    use_idf=True,
                                    sublinear_tf=True)

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[str]]:        
        # Extract keywords using Tf-Idf entities
        entities = []
        for abstract in abstracts:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([abstract])
            keywords = self.tfidf_vectorizer.get_feature_names_out()
            entities.append([(ent, 1.0) for ent in keywords])
        
        return entities