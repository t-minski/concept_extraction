from yake import KeywordExtractor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from itertools import product
from models.base_model import BaseModel
from typing import List, Tuple

# Set up the logger
import logging
logger = logging.getLogger(__name__)

class YakeEntities(BaseModel):
    
    def __init__(self):
        self.kw_extractor = KeywordExtractor(lan="en")
        self.best_params = None

    def evaluate(self, abstracts: List[str], concepts: List[List[str]], model) -> float:
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        num_docs = len(abstracts)

        for i, abstract in enumerate(abstracts):
            extracted_keywords = model.predict([abstract])[0]
            extracted_keywords = [kw for kw, score in extracted_keywords]
            true_keywords = concepts[i]

            # Create binary vectors for precision_recall_fscore_support
            y_true = [1 if kw in true_keywords else 0 for kw in extracted_keywords]
            y_pred = [1 for _ in extracted_keywords]

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
        # Define the hyperparameter grid for YAKE
        param_grid = {
            'lan': ['en'],
            'n': [1, 2, 3],
            'dedupLim': [0.5, 0.6, 0.7, 0.8, 0.9],
            'dedupFunc': ['seqm', 'levs'],
            'windowsSize': [1, 2, 3],
            'top': [10, 20, 30]
        }

        best_score = 0
        best_params = None

        # Generate all possible combinations of hyperparameters
        for params in product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), params))
            
            # Set parameters for YAKE
            self.kw_extractor = KeywordExtractor(**param_dict)
            
            # Evaluate the model
            score = self.evaluate(abstracts, concepts, self)

            if score > best_score:
                best_score = score
                best_params = param_dict

        logger.info("hyperparameter optimization")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best F1 score: {best_score}")
        self.best_params = best_params
        # Update the keyword extractor with the best parameters
        self.kw_extractor = KeywordExtractor(**self.best_params)

    def predict(self, abstracts: List[str]) -> List[List[str]]:        
        # Extract keywords using YAKE entities
        entities = []
        for abstract in abstracts:
            yake_keywords = self.kw_extractor.extract_keywords(abstract)
            keywords_with_scores = [(keyword[0], keyword[1]) for keyword in yake_keywords]
            entities.append(keywords_with_scores)
        
        return entities