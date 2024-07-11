from conexion.models.base_model import BaseModel
from typing import List, Tuple
import json

class ResultsLoaderModel(BaseModel):

    def __init__(self, file_name:str):
        self.file_name = file_name
        #remove the file extension
        self.model_template_name = file_name.rsplit("/", 1)[-1]
        self.model_template_name = self.model_template_name.rsplit(".", 1)[0]

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[Tuple[str, float]]]:
        # parse the file with csv and extract "Extracted Keywords" column
        import csv
        with open(self.file_name, 'r') as file:
            reader = csv.reader(file)
            reader.__next__()  # skip the header
            extracted_keywords = [json.loads(row[1].replace("'", '"')) for row in reader]
        return extracted_keywords