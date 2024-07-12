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
        import ast
        extracted_keywords = []
        with open(self.file_name, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            reader.__next__()  # skip the header
            for row in reader:
                try:
                    keywords = ast.literal_eval(row[1])
                    if isinstance(keywords, list):
                        extracted_keywords.append([(keyword, 1.0) for keyword in keywords])
                    else:
                        print(f"Warning: row {row} does not contain a list in the expected column.")
                except (ValueError, SyntaxError) as e:
                    print(f"Error parsing keywords in row {row}: {e}")        
        return extracted_keywords