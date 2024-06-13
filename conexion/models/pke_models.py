import pke
from models.base_model import BaseModel
from typing import List, Tuple
from pke import compute_document_frequency, compute_lda_model
from string import punctuation
from pke import load_document_frequency_file, load_lda_model
from pke import train_supervised_model
from nltk.stem.snowball import SnowballStemmer as Stemmer
import os



class Kea(BaseModel):

    def __init__(self):
        self.method = 'Kea'

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        df_file = "data/{}.df.gz".format("benchmark")
        if not os.path.exists(df_file):           
            compute_document_frequency(
                documents=abstracts,
                output_file=df_file, 
                language='en',                   # language of the input files
                normalization='stemming',        # use porter stemmer
                stoplist=list(punctuation),      # stoplist (punctuation marks)
                n=5                              # compute n-grams up to 5-grams
            )
        self.df = load_document_frequency_file(input_file=df_file)
        
        self.kea_file = "data/{}.kea.model.pickle".format("benchmark")
        if not os.path.exists(self.kea_file):
                samples = []
                references = {}
                for sample in abstracts:
                    samples.append((sample["id"], train[len(samples)]))
                    references[sample["id"]] = []
                    for keyphrase in concepts:
                        # tokenize keyphrase
                        tokens = [token.text for token in nlp(keyphrase)]
                        # normalize tokens using Porter's stemming
                        stems = [Stemmer('porter').stem(tok.lower()) for tok in tokens]
                        references[sample["id"]].append(" ".join(stems))
                
            train_supervised_model(
                documents=abstracts,
                references=references,
                model_file=self.kea_file,
                language='en',
                normalization='stemming',
                df=self.df,
                model=pke.supervised.Kea()
            )

    def predict(self, abstracts: List[str]) -> List[List[Tuple[str, float]]]:
        entities = []
        for abstract in abstracts:
            extractor = pke.unsupervised.__dict__[self.method]()
            extractor.load_document(input=abstract, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting(df=self.df, model_file=self.kea_file))
            keywords = extractor.get_n_best(n=1000000, stemming=False)  # Set a very high number to get all keywords
            entities.append(keywords)
        
        return entities

class KPMiner(BaseModel):

    def __init__(self):
        self.method = 'KPMiner'

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        df_file = "data/{}.df.gz".format("benchmark")
        if not os.path.exists(df_file):           
            compute_document_frequency(
                documents=abstracts,
                output_file=df_file, 
                language='en',                   # language of the input files
                normalization='stemming',        # use porter stemmer
                stoplist=list(punctuation),      # stoplist (punctuation marks)
                n=5                              # compute n-grams up to 5-grams
            )
        self.df = load_document_frequency_file(input_file=df_file)

    def predict(self, abstracts: List[str]) -> List[List[Tuple[str, float]]]:
        entities = []
        for abstract in abstracts:
            extractor = pke.unsupervised.__dict__[self.method]()
            extractor.load_document(input=abstract, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting(df=self.df)
            keywords = extractor.get_n_best(n=1000000, stemming=False)  # Set a very high number to get all keywords
            entities.append(keywords)
        
        return entities

class TfIdf(BaseModel):

    def __init__(self):
        self.method = 'TfIdf'

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        df_file = "data/{}.df.gz".format("benchmark")
        if not os.path.exists(df_file):           
            compute_document_frequency(
                documents=abstracts,
                output_file=df_file, 
                language='en',                   # language of the input files
                normalization='stemming',        # use porter stemmer
                stoplist=list(punctuation),      # stoplist (punctuation marks)
                n=5                              # compute n-grams up to 5-grams
            )
        self.df = load_document_frequency_file(input_file=df_file)

    def predict(self, abstracts: List[str]) -> List[List[Tuple[str, float]]]:
        entities = []
        for abstract in abstracts:
            extractor = pke.unsupervised.__dict__[self.method]()
            extractor.load_document(input=abstract, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting(df=self.df)
            keywords = extractor.get_n_best(n=1000000, stemming=False)  # Set a very high number to get all keywords
            entities.append(keywords)
        
        return entities

class YAKE(BaseModel):

    def __init__(self):
        self.method = 'YAKE'

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[Tuple[str, float]]]:
        entities = []
        for abstract in abstracts:
            extractor = pke.unsupervised.__dict__[self.method]()
            extractor.load_document(input=abstract, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keywords = extractor.get_n_best(n=1000000, stemming=False)  # Set a very high number to get all keywords
            entities.append(keywords)
        
        return entities

class TopicRank(BaseModel):

    def __init__(self):
        self.method = 'TopicRank'

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[Tuple[str, float]]]:
        entities = []
        for abstract in abstracts:
            extractor = pke.unsupervised.__dict__[self.method]()
            extractor.load_document(input=abstract, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keywords = extractor.get_n_best(n=1000000, stemming=False)  # Set a very high number to get all keywords
            entities.append(keywords)
        
        return entities

class MultipartiteRank(BaseModel):

    def __init__(self):
        self.method = 'MultipartiteRank'

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[Tuple[str, float]]]:
        entities = []
        for abstract in abstracts:
            extractor = pke.unsupervised.__dict__[self.method]()
            extractor.load_document(input=abstract, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keywords = extractor.get_n_best(n=1000000, stemming=False)  # Set a very high number to get all keywords
            entities.append(keywords)
        
        return entities

class TopicalPageRank(BaseModel):

    def __init__(self):
        self.method = 'TopicalPageRank'

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[Tuple[str, float]]]:
        entities = []
        for abstract in abstracts:
            extractor = pke.unsupervised.__dict__[self.method]()
            extractor.load_document(input=abstract, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keywords = extractor.get_n_best(n=1000000, stemming=False)  # Set a very high number to get all keywords
            entities.append(keywords)
        
        return entities

class PositionRank(BaseModel):

    def __init__(self):
        self.method = 'PositionRank'

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[Tuple[str, float]]]:
        entities = []
        for abstract in abstracts:
            extractor = pke.unsupervised.__dict__[self.method]()
            extractor.load_document(input=abstract, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keywords = extractor.get_n_best(n=1000000, stemming=False)  # Set a very high number to get all keywords
            entities.append(keywords)
        
        return entities