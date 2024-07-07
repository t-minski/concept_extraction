import pke
from conexion.models.base_model import BaseModel
from typing import List, Tuple
from pke import compute_document_frequency, compute_lda_model
from string import punctuation
from pke import load_document_frequency_file, load_lda_model
from pke import train_supervised_model
from nltk.stem.snowball import SnowballStemmer as Stemmer
import os
import spacy
from spacy.tokenizer import _get_regex_pattern

nlp = spacy.load("en_core_web_sm", disable=['ner', 'textcat', 'parser'])
nlp.add_pipe("sentencizer")
# Tokenization fix for in-word hyphens (e.g. 'non-linear' would be kept 
# as one token instead of default spacy behavior of 'non', '-', 'linear')
# https://spacy.io/usage/linguistic-features#native-tokenizer-additions

from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

# Modify tokenizer infix patterns
infixes = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        r"(?<=[0-9])[+\-\*^](?=[0-9-])",
        r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
            al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
        ),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        # ✅ Commented out regex that splits on hyphens between letters:
        # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
        r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
    ]
)

infix_re = compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_re.finditer

class pke_FirstPhrases(BaseModel):

    def __init__(self):
        self.method = 'FirstPhrases'

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[Tuple[str, float]]]:
        entities = []
        for abstract in abstracts:
            extractor = pke.unsupervised.__dict__[self.method]()
            extractor.load_document(input=abstract, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keywords = extractor.get_n_best(n=20, stemming=False)  # Set a very high number to get all keywords
            entities.append(keywords)
        
        return entities

class pke_TextRank(BaseModel):

    def __init__(self):
        self.method = 'TextRank'

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[Tuple[str, float]]]:
        entities = []
        for abstract in abstracts:
            extractor = pke.unsupervised.__dict__[self.method]()
            extractor.load_document(input=abstract, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keywords = extractor.get_n_best(n=20, stemming=False)  # Set a very high number to get all keywords
            entities.append(keywords)
        
        return entities
    

class pke_SingleRank(BaseModel):

    def __init__(self):
        self.method = 'SingleRank'

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        pass

    def predict(self, abstracts: List[str]) -> List[List[Tuple[str, float]]]:
        entities = []
        for abstract in abstracts:
            extractor = pke.unsupervised.__dict__[self.method]()
            extractor.load_document(input=abstract, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keywords = extractor.get_n_best(n=20, stemming=False)  # Set a very high number to get all keywords
            entities.append(keywords)
        
        return entities


class pke_Kea(BaseModel):

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
            train = abstracts
            samples = []
            references = {}
            for sample_id, sample in enumerate(abstracts):
                samples.append((sample_id, train[len(samples)]))
                references[sample_id] = []
                for keyphrase in concepts[sample_id]:
                    # tokenize keyphrase
                    tokens = [token.text for token in nlp(keyphrase)]
                    # normalize tokens using Porter's stemming
                    stems = [Stemmer('porter').stem(tok.lower()) for tok in tokens]
                    references[sample_id].append(" ".join(stems))
                
            train_supervised_model(
                documents=samples,
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
            extractor = pke.supervised.__dict__[self.method]()
            extractor.load_document(input=abstract, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting(df=self.df, model_file=self.kea_file)
            keywords = extractor.get_n_best(n=20, stemming=False)  # Set a very high number to get all keywords
            entities.append(keywords)
        
        return entities

class pke_KPMiner(BaseModel):

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
            keywords = extractor.get_n_best(n=20, stemming=False)  # Set a very high number to get all keywords
            entities.append(keywords)
        
        return entities

class pke_TfIdf(BaseModel):

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
            keywords = extractor.get_n_best(n=20, stemming=False)  # Set a very high number to get all keywords
            entities.append(keywords)
        
        return entities

class pke_YAKE(BaseModel):

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
            keywords = extractor.get_n_best(n=20, stemming=False)  # Set a very high number to get all keywords
            entities.append(keywords)
        
        return entities

class pke_TopicRank(BaseModel):

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
            keywords = extractor.get_n_best(n=20, stemming=False)  # Set a very high number to get all keywords
            entities.append(keywords)
        
        return entities

class pke_MultipartiteRank(BaseModel):

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
            keywords = extractor.get_n_best(n=20, stemming=False)  # Set a very high number to get all keywords
            entities.append(keywords)
        
        return entities

class pke_TopicalPageRank(BaseModel):

    def __init__(self):
        self.method = 'TopicalPageRank'

    def fit(self, abstracts: List[str], concepts: List[List[str]]) -> None:
        lda_file = "data/{}.lda.pickle.gz".format("benchmark")
        if not os.path.exists(lda_file):
            compute_lda_model(
                documents=abstracts,
                output_file=lda_file,
                n_topics=20,              # number of topics
                language='en',              # language of the input files
                stoplist=list(punctuation), # stoplist (punctuation marks)
                normalization='stemming'    # use porter stemmer
            )
        self.lda_model = load_lda_model(input_file=lda_file)

    def predict(self, abstracts: List[str]) -> List[List[Tuple[str, float]]]:
        entities = []
        for abstract in abstracts:
            extractor = pke.unsupervised.__dict__[self.method]()
            extractor.load_document(input=abstract, language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting(lda_model=self.lda_model)
            keywords = extractor.get_n_best(n=20, stemming=False)  # Set a very high number to get all keywords
            entities.append(keywords)
        
        return entities

class pke_PositionRank(BaseModel):

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
            keywords = extractor.get_n_best(n=20, stemming=False)  # Set a very high number to get all keywords
            entities.append(keywords)
        
        return entities