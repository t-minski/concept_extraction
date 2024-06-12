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
            # Fit and transform the abstract to get the TF-IDF matrix
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([abstract])
            
            # Get the feature names (keywords) and their corresponding TF-IDF scores
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Create a list of tuples (keyword, score)
            keywords_with_scores = [(feature_names[idx], tfidf_scores[idx]) for idx in tfidf_scores.argsort()[::-1]]
            
            # Append the keywords with their scores to the entities list
            entities.append(keywords_with_scores)
    
        return entities