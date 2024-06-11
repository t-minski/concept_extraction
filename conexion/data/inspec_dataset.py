from data.base_dataset import BaseDataset
from typing import List, Tuple

from datasets import load_dataset

class Inspec(BaseDataset):                
    def get_training_data(self) -> Tuple[List[str], List[List[str]]]:
        training_dataset = load_dataset("taln-ls2n/inspec", split="train")
        return training_dataset["abstract"], training_dataset["keyphrases"]
    
    def get_test_data(self) -> Tuple[List[str], List[List[str]]]:
        test_dataset = load_dataset("taln-ls2n/inspec", split="test")
        return test_dataset["abstract"], test_dataset["keyphrases"]