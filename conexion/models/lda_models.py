from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import preprocess_string, STOPWORDS
from sklearn.metrics import precision_recall_fscore_support
from itertools import product
from models.base_model import BaseModel
from typing import List, Tuple

# Set up the logger
import logging
logger = logging.getLogger(__name__)

class LDAEntities(BaseModel):
    
    def __init__(self):
        self.num_topics = 1  # Adjust the number of topics as needed

    def evaluate(self, abstracts: List[str], concepts: List[List[str]], num_topics: int) -> float:
        # Prepare the data
        texts = [[token for token in preprocess_string(abstract) if token not in STOPWORDS] for abstract in abstracts]
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, random_state=42, alpha='auto')

        total_precision = 0
        total_recall = 0
        total_f1 = 0
        num_docs = len(abstracts)

        for doc_idx, abstract in enumerate(abstracts):
            bow = dictionary.doc2bow(abstract.split())
            topics = lda_model.get_document_topics(bow)
            
            extracted_keywords = []
            for topic_id, topic_prob in topics:
                topic_keywords = lda_model.show_topic(topic_id)
                filtered_keywords = [word for word, prob in topic_keywords]# if word in abstract]
                extracted_keywords.extend(filtered_keywords)
            
            # Remove duplicates while preserving order
            seen = set()
            extracted_keywords = [k for k in extracted_keywords if not (k in seen or seen.add(k))]

            true_keywords = concepts[doc_idx]

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
        logger.info(f"extracted_keywords {extracted_keywords}, true_keywords: {true_keywords}")
        logger.info(f"Evaluation - num_topics: {num_topics}, F1 score: {avg_f1}")

        return avg_f1

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        # Define the hyperparameter grid for LDA
        param_grid = {
            'num_topics': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }

        best_score = 0
        best_params = None

        # Generate all possible combinations of hyperparameters
        for params in product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), params))
            score = self.evaluate(abstracts, concepts, param_dict['num_topics'])

            if score > best_score:
                best_score = score
                best_params = param_dict

        if best_params is None:
            logger.error("No valid parameters found during hyperparameter optimization.")
            self.num_topics = 1
        else:
            logger.info("hyperparameter optimization")
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best F1 score: {best_score}")
            self.num_topics = best_params['num_topics']
        
        # Train the final model with the best parameters
        texts = [[token for token in preprocess_string(abstract) if token not in STOPWORDS] for abstract in abstracts]
        self.dictionary = Dictionary(texts)
        corpus = [self.dictionary.doc2bow(text) for text in texts]

        self.lda_model = LdaModel(corpus, num_topics=self.num_topics, id2word=self.dictionary, random_state=42, alpha='auto')


        
    def predict(self, abstracts: List[str]) -> List[List[Tuple[str, float]]]:
        if not self.dictionary or not self.lda_model:
            raise ValueError("The model has not been trained. Please call the fit method first.")
        
        # Extract keywords using LDA entities
        entities = []
        for abstract in abstracts:
            preprocessed_abstract = [token for token in preprocess_string(abstract) if token not in STOPWORDS]
            bow = self.dictionary.doc2bow(preprocessed_abstract)
            topics = self.lda_model.get_document_topics(bow)
            
            keywords_with_scores = []
            for topic_id, topic_prob in topics:
                topic_keywords = self.lda_model.show_topic(topic_id)
                # Include only keywords that are in the original preprocessed abstract
                filtered_keywords = [(word, prob) for word, prob in topic_keywords if word in preprocessed_abstract]
                keywords_with_scores.extend(filtered_keywords)
            
            # Remove duplicates while preserving order
            seen = set()
            keywords_with_scores = [(k, v) for k, v in keywords_with_scores if not (k in seen or seen.add(k))]
            
            entities.append(keywords_with_scores)
            
        return entities