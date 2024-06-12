from data.base_dataset import BaseDataset
from typing import List, Tuple

from datasets import load_dataset

class semeval2017(BaseDataset):                
    def get_training_data(self) -> Tuple[List[str], List[List[str]]]:
        training_dataset = load_dataset("midas/semeval2017", "raw")["train"]
        return [' '.join(tokens) for tokens in training_dataset["document"]], training_dataset["extractive_keyphrases"]
    
    def get_test_data(self) -> Tuple[List[str], List[List[str]]]:
        test_dataset = load_dataset("midas/semeval2017", "raw")["test"]
        return [' '.join(tokens) for tokens in test_dataset["document"]], test_dataset["extractive_keyphrases"]