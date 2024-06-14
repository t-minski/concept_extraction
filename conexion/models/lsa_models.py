from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import precision_recall_fscore_support
from itertools import product
from models.base_model import BaseModel
from typing import List, Tuple

# Set up the logger
import logging
logger = logging.getLogger(__name__)

class LSAEntities(BaseModel):
    
    def __init__(self):
        self.lsa_model = TruncatedSVD(n_components=10)  # Adjust the number of components as needed
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    def evaluate(self, abstracts: List[str], concepts: List[List[str]], vectorizer: TfidfVectorizer, svd_model: TruncatedSVD) -> float:
        # Transform the abstracts using the given vectorizer
        tfidf_matrix = vectorizer.fit_transform(abstracts)
        lsa_matrix = svd_model.fit_transform(tfidf_matrix)
        feature_names = vectorizer.get_feature_names_out()

        total_precision = 0
        total_recall = 0
        total_f1 = 0
        num_docs = len(abstracts)

        for doc_idx in range(num_docs):
            # Extract keywords based on the components
            component_scores = svd_model.components_[0]
            sorted_indices = component_scores.argsort()[::-1]
            extracted_keywords = [feature_names[i] for i in sorted_indices]

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
            'tfidf__min_df': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'tfidf__norm': ['l1', 'l2', None],
            'tfidf__use_idf': [True, False],
            'tfidf__sublinear_tf': [True, False],
            'svd__n_components': [10, 20, 30, 40, 50]  # Ensure these are within a reasonable range
        }

        best_score = 0
        best_params = None
        best_vectorizer = None
        best_svd_model = None

        # Generate all possible combinations of hyperparameters
        for params in product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), params))
            
            # Create vectorizer and transform to check number of features
            vectorizer = TfidfVectorizer(
                min_df=param_dict['tfidf__min_df'],
                ngram_range=param_dict['tfidf__ngram_range'],
                norm=param_dict['tfidf__norm'],
                use_idf=param_dict['tfidf__use_idf'],
                sublinear_tf=param_dict['tfidf__sublinear_tf'],
                stop_words='english'
            )
            tfidf_matrix = vectorizer.fit_transform(abstracts)
            n_features = tfidf_matrix.shape[1]

            # Adjust n_components to be <= n_features
            n_components = min(param_dict['svd__n_components'], n_features)
            
            svd_model = TruncatedSVD(n_components=n_components)
            score = self.evaluate(abstracts, concepts, vectorizer, svd_model)

            if score > best_score:
                best_score = score
                best_params = param_dict
                best_vectorizer = vectorizer
                best_svd_model = svd_model

        logger.info("hyperparameter optimization")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best F1 score: {best_score}")
        self.tfidf_vectorizer = best_vectorizer
        self.lsa_model = best_svd_model

    def predict(self, abstracts: List[str]) -> List[List[str]]:        
        # Extract keywords using LSA entities
        entities = []
        for abstract in abstracts:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([abstract])
            tfidf_keywords = self.tfidf_vectorizer.get_feature_names_out()
            lsa_matrix = self.lsa_model.fit_transform(tfidf_matrix)
            
            # Get the components and their scores
            components = self.lsa_model.components_[0]
            keywords_with_scores = [(tfidf_keywords[i], components[i]) for i in components.argsort()[::-1]]
            
            entities.append(keywords_with_scores)
        
        return entities