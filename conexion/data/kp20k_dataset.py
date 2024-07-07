from conexion.data.base_dataset import BaseDataset
from conexion.data.statistics_logger import log_statistics
from typing import List, Tuple
from datasets import load_dataset

class kp20k(BaseDataset):                
    def get_training_data(self) -> Tuple[List[str], List[List[str]]]:
        training_dataset = load_dataset("midas/kp20k", "generation", split="train", revision="a871c01", trust_remote_code=True)
        filtered_documents, filtered_keyphrases = log_statistics(
                                                                    training_dataset["document"],
                                                                    training_dataset["extractive_keyphrases"]
                                                                )

        return filtered_documents, filtered_keyphrases
        
    def get_test_data(self) -> Tuple[List[str], List[List[str]]]:
        test_dataset = load_dataset("midas/kp20k", "generation", split="test", revision="a871c01", trust_remote_code=True)
        filtered_documents, filtered_keyphrases = log_statistics(
                                                                    test_dataset["document"],
                                                                    test_dataset["extractive_keyphrases"]
                                                                )

        return filtered_documents, filtered_keyphrases