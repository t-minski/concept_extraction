import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])

packages = [
    "git+https://github.com/UKPLab/sentence-transformers",
    "keybert",
    "ctransformers[cuda]",
    "git+https://github.com/huggingface/transformers",
    "spacy",
    "yake",
    "gensim",
    "pyate",
    "rake-nltk",
    "summa",
    "keybert",
    "huggingface_hu==0.10.1",
    "bibtexparser",
    "Levenshtein",
    "fuzzywuzzy"
]

for package in packages:
    install(package)

subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
