from typing import List, Tuple

class BaseModel:

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        """Trains the model on the given abstracts and concepts.
        Parameters
        ----------
        abstracts: List of strings
            List of abstracts to train the models on.
        concepts: List of lists of strings
            Concepts corresponding to each abstract
        Returns
        ----------"""
        pass

    def predict(self, abstracts: List[str]) -> List[List[Tuple[str, float]]]:
        """Predicts the concepts for the given abstracts.
        Parameters
        ----------
        abstracts: List of strings
            List of abstracts to predict the concepts for.
        Returns
        ----------
        List of lists of strings
            Concepts corresponding to each abstract
        """
        pass