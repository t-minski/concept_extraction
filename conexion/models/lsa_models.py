from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from models.base_model import BaseModel
from typing import List, Tuple

class LSAEntities(BaseModel):
    
    def __init__(self):
        self.lsa_model = TruncatedSVD(n_components=10)  # Adjust the number of components as needed
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
        # Extract keywords using LSA entities
        entities = []
        for abstract in abstracts:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([abstract])
            tfidf_keywords = self.tfidf_vectorizer.get_feature_names_out()
            lsa_matrix = self.lsa_model.fit_transform(tfidf_matrix)
            keywords = [tfidf_keywords[i] for i in self.lsa_model.components_[0].argsort()[::-1]]
            entities.append([(ent, 1.0) for ent in keywords])
        
        return entities