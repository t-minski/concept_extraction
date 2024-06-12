from data.base_dataset import BaseDataset
from typing import List, Tuple

from datasets import load_dataset

class semeval2010(BaseDataset):                
    def get_training_data(self) -> Tuple[List[str], List[List[str]]]:
        training_dataset = load_dataset("midas/semeval2010", "raw")["train"]
        filtered_documents = [' '.join(tokens) for tokens in training_dataset["document"]]
        filtered_keyphrases = [[phrase for phrase in keyphrases if phrase in ' '.join(tokens)]
                            for tokens, keyphrases in zip(training_dataset["document"], training_dataset["extractive_keyphrases"])]

        return filtered_documents, filtered_keyphrases
        
    def get_test_data(self) -> Tuple[List[str], List[List[str]]]:
        test_dataset = load_dataset("midas/semeval2010", "raw")["test"]
        filtered_documents = [' '.join(tokens) for tokens in test_dataset["document"]]
        filtered_keyphrases = [[phrase for phrase in keyphrases if phrase in ' '.join(tokens)]
                            for tokens, keyphrases in zip(test_dataset["document"], test_dataset["extractive_keyphrases"])]

        return filtered_documents, filtered_keyphrases