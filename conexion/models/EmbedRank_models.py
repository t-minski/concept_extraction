import spacy
from spacycake import BertKeyphraseExtraction as bake
from models.base_model import BaseModel
from typing import List, Tuple

class EmbedRank(BaseModel):
    
    def __init__(self):
        # Load the SpaCy model
        self.nlp = spacy.load('en')
        # Initialize the BERT keyphrase extraction
        self.cake = bake(self.nlp, from_pretrained='bert-base-cased', top_k=10)

    def fit(self, abstracts: List[str], keyphrases: List[List[str]]) -> None:
        # No fitting needed for this model
        pass

    def predict(self, abstracts: List[str]) -> List[List[str]]:        
        entities = []
        for abstract in abstracts:
            # Convert the text to a SpaCy Doc object
            doc = self.nlp(abstract)
            # Extract keyphrases
            keywords = self.cake._get_candidate_phrases(doc)[:10]
            keywords_with_scores = [
                                    (str(keyword).rstrip('.'), 1.0)  # Assigning score 1 to all keywords
                                    for keyword in keywords 
                                    if str(keyword) in abstract.lower()
                                    ]
            entities.append(keywords_with_scores)
        
        return entities