from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from itertools import product
from conexion.models.base_model import BaseModel
from typing import List, Tuple

# Set up the logger
import logging
logger = logging.getLogger(__name__)

class TfIdfEntities(BaseModel):
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english',)
        self.tfidf_threshold = 0.1  # Set a default TF-IDF threshold

    def evaluate(self, abstracts: List[str], concepts: List[List[str]], vectorizer: TfidfVectorizer) -> float:
        # Transform the abstracts using the given vectorizer
        tfidf_matrix = vectorizer.fit_transform(abstracts)
        feature_names = vectorizer.get_feature_names_out()

        total_precision = 0
        total_recall = 0
        total_f1 = 0
        num_docs = len(abstracts)

        for doc_idx, doc in enumerate(tfidf_matrix):
            extracted_keywords = [feature_names[i] for i in doc.indices]
            true_keywords = concepts[doc_idx]

            # Create binary vectors for precision_recall_fscore_support
            y_true = [1 if kw in true_keywords else 0 for kw in feature_names]
            y_pred = [1 if kw in extracted_keywords else 0 for kw in feature_names]

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )

            total_precision += precision
            total_recall += recall
            total_f1 += f1

        avg_precision = total_precision / num_docs
        avg_recall = total_recall / num_docs
        avg_f1 = total_f1 / num_docs

        return avg_f1

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        # Define the hyperparameter grid
        param_grid = {
            'min_df': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], #0.01 means ignore terms that appear in less than 1% of the documents
            #'max_df': [0.7, 0.8, 0.9, 1.0], #0.50 means ignore terms that appear in more than 50% of the documents
            'ngram_range': [(1, 1), (1, 2), (1, 3)],
            'norm': ['l1', 'l2', None],
            'use_idf': [True, False],
            'sublinear_tf': [True, False]
        }

        best_score = 0
        best_params = None
        best_vectorizer = None

        # Generate all possible combinations of hyperparameters
        for params in product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), params))
            vectorizer = TfidfVectorizer(**param_dict)
            score = self.evaluate(abstracts, concepts, vectorizer)

            if score > best_score:
                best_score = score
                best_params = param_dict
                best_vectorizer = vectorizer

        logger.info("hyperparameter optimization")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best F1 score: {best_score}")
        self.tfidf_vectorizer = best_vectorizer

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