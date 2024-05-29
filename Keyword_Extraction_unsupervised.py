# %%
%%capture
!pip install spacy
!pip install yake
!pip install gensim
!pip install pyate
!pip install rake-nltk
!pip install summa
!pip install Levenshtein
!pip install fuzzywuzzy

# %%
import csv
import os
from yake import KeywordExtractor
from rake_nltk import Rake
from yake import KeywordExtractor
from rake_nltk import Rake
from pyate import combo_basic, basic, cvalues
from summa import keywords as summa_keywords
import spacy
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
import Levenshtein
# %%
import nltk
nltk.download('stopwords')
spacy.load("en_core_web_lg")
# %%
def extract_keywords_from_abstract(abstract):

    # Get the English stopwords
    stop_words = set(stopwords.words('english'))
    abstract = ' '.join([word for word in abstract.split() if word.lower() not in stop_words])

    # Initialize Spacy, YAKE, and RAKE keyword extractors
    nlp = spacy.load("en_core_web_lg")
    kw_extractor = KeywordExtractor() # KeywordExtractor(lan="en", n=3, dedupLim=0.6, dedupFunc='seqm', windowsSize=1, top=20, features=None)
    rake_nltk_var = Rake()

    # Extract keywords using Spacy entities
    doc = nlp(abstract)
    spacy_entities = [ent.text for ent in doc.ents]

    # Extract keywords using Spacy noun chunks
    doc = nlp(abstract)
    spacy_noun_chunks = [chunk.text for chunk in doc.noun_chunks]

    # Extract keywords using YAKE
    yake_keywords = kw_extractor.extract_keywords(abstract)
    yake_keywords = [keyword[0] for keyword in yake_keywords]

    # Extract keywords using RAKE
    rake_nltk_var.extract_keywords_from_text(abstract)
    rake_keywords = rake_nltk_var.get_ranked_phrases()

    # Extract keywords using Pyate
    pyate_combo_basic_keywords = combo_basic(abstract).sort_values(ascending=False).index.str.split().str[0].tolist()
    pyate_basic_keywords = basic(abstract).sort_values(ascending=False).index.str.split().str[0].tolist()
    pyate_cvalues_keywords = cvalues(abstract).sort_values(ascending=False).index.str.split().str[0].tolist()

    # Extract keywords using summa
    summa_keywords_ = [keyword[0] for keyword in summa_keywords.keywords(abstract, scores=True)]
    
    # Extract keywords using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    tfidf_matrix = tfidf_vectorizer.fit_transform([abstract])
    tfidf_keywords = tfidf_vectorizer.get_feature_names_out()

    # Extract keywords using LSA
    lsa_model = TruncatedSVD(n_components=10)  # Adjust the number of components as needed
    lsa_matrix = lsa_model.fit_transform(tfidf_matrix)
    lsa_keywords = [tfidf_keywords[i] for i in lsa_model.components_[0].argsort()[::-1]]

    # Extract keywords using LDA
    dictionary = Dictionary([abstract.split()])
    corpus = [dictionary.doc2bow(abstract.split())]
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in [abstract.split()]]
    lda_model = LdaModel(corpus=doc_term_matrix, num_topics=10, id2word=dictionary)  # Adjust the number of topics as needed
    lda_keywords = [word for word, _ in lda_model.show_topic(0)]

    return {
        "Spacy_entities": spacy_entities,
        "Spacy_noun_chunks": spacy_noun_chunks,
        "YAKE_keywords": yake_keywords,
        "RAKE_keywords": rake_keywords,
        "Pyate_combo_basic_keywords": pyate_combo_basic_keywords,
        "Pyate_basic_keywords": pyate_basic_keywords,
        "Pyate_cvalues_keywords": pyate_cvalues_keywords,
        "Summa_keywords": summa_keywords_,
        "TFIDF_keywords": tfidf_keywords,
        "LSA_keywords": lsa_keywords,
        "LDA_keywords": lda_keywords,
    }


# %%
# Function to tokenize and stem text
def tokenize_and_stem(text):
    stemmer = PorterStemmer()
    if isinstance(text, str):
        tokens = [stemmer.stem(word) for word in text.split()]
        return ' '.join(tokens)
    else:
        return str(text)

# Function to evaluate keywords with Levenshtein threshold
def evaluate_keywords(ground_truth_keywords, extracted_keywords, threshold=0.8):
    ground_truth_keywords = list(set(ground_truth_keywords))
    extracted_keywords = list(set(extracted_keywords))
    
    # Initialize variables for evaluation metrics
    tp, fp, fn = 0, 0, 0

    # Tokenize and stem ground truth keywords
    ground_truth_stems = [tokenize_and_stem(keyword) for keyword in ground_truth_keywords]

    # Iterate over extracted keywords
    for extracted_keyword in extracted_keywords:
        # Tokenize and stem extracted keyword
        extracted_stem = tokenize_and_stem(extracted_keyword)

        # Check if extracted keyword matches any ground truth keyword within Levenshtein threshold
        matched = False
        for ground_truth_stem in ground_truth_stems:
            max_len = max(len(extracted_stem), len(ground_truth_stem))
            if Levenshtein.distance(extracted_stem, ground_truth_stem) / max_len <= 1 - threshold:
                matched = True
                break

        # Update evaluation metrics based on match status
        if matched:
            tp += 1
        else:
            fp += 1

    # Calculate false negatives (missed ground truth keywords)
    fn = len(ground_truth_keywords) - tp

    # Calculate precision, recall, and F1-score
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return precision, recall, f1_score

# %%
def read_files_from_directory(base_path, dataset):
    abstracts = []
    keywords = []

    docs_path = os.path.join(base_path, dataset, 'docsutf8')
    keys_path = os.path.join(base_path, dataset, 'keys')

    for filename in os.listdir(docs_path):
        if filename.endswith('.txt'):
            with open(os.path.join(docs_path, filename), 'r', encoding='utf-8') as file:
                if dataset == 'SemEval2010':
                    content = file.read()
                    abstract = extract_semeval_abstract(content)
                else:
                    abstract = file.read().strip()
                #abstracts[identifier] = abstract
                abstracts.append(abstract)

    for filename in os.listdir(keys_path):
        if filename.endswith('.key'):
            with open(os.path.join(keys_path, filename), 'r', encoding='utf-8') as file:
                keywords.append([line.strip() for line in file.readlines()])
    return abstracts, keywords

def extract_semeval_abstract(content):
    lines = content.split('\n')
    abstract_lines = []
    in_abstract = False
    for line in lines:
        if line.startswith('ABSTRACT'):
            in_abstract = True
        elif line.startswith('Categories and Subject Descriptors'):
            in_abstract = False
        elif in_abstract:
            abstract_lines.append(line.strip())
    return ' '.join(abstract_lines).strip()

# %%
def evaluate_keywords_from_data(base_path, datasets, extraction_functions, output_folder):
    cumulative_precision = {method: 0 for method in extraction_functions}
    cumulative_recall = {method: 0 for method in extraction_functions}
    cumulative_f1_score = {method: 0 for method in extraction_functions}

    all_evaluation_results = []
    all_evaluation_results_avg = []
    total_abstracts = 0
    for dataset in datasets:
        abstracts, keywords = read_files_from_directory(base_path, dataset)
        

        #for identifier, abstract in abstracts.items():
        for abstract, ground_truth_keywords in zip(abstracts, keywords):

            total_abstracts += 1

            for method, extraction_function in extraction_functions.items():
                extracted_keywords = extraction_function(abstract.lower())
                precision, recall, f1_score = evaluate_keywords(ground_truth_keywords, extracted_keywords)

                cumulative_precision[method] += precision
                cumulative_recall[method] += recall
                cumulative_f1_score[method] += f1_score

                all_evaluation_results.append((ground_truth_keywords, extracted_keywords, method, precision, recall, f1_score, len(ground_truth_keywords), len(extracted_keywords)))
    
    average_precision = {method: cumulative_precision[method] / total_abstracts for method in extraction_functions}
    average_recall = {method: cumulative_recall[method] / total_abstracts for method in extraction_functions}
    average_f1_score = {method: cumulative_f1_score[method] / total_abstracts for method in extraction_functions}

    print("Average Scores over all Abstracts:")
    for method in extraction_functions:
        print(f"{method}, {average_precision[method]}, {average_recall[method]}, {average_f1_score[method]}")
        all_evaluation_results_avg.append((method, average_precision[method], average_recall[method], average_f1_score[method]))

    with open(os.path.join(output_folder, 'evaluation_results.csv'), 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Ground_truth Keywords', 'Extracted Keywords', 'Method', 'Precision', 'Recall', 'F1-score', 'n_gt_keywords', 'n_extracted_keywords'])
        writer.writerows(all_evaluation_results)

    with open(os.path.join(output_folder, 'evaluation_results_avg.csv'), 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Method', 'Precision', 'Recall', 'F1-score'])
        writer.writerows(all_evaluation_results_avg)


# %%
# Define extraction functions
extraction_functions = {
    "Spacy_entities": lambda abstract: extract_keywords_from_abstract(abstract)["Spacy_entities"],
    "Spacy_noun_chunks": lambda abstract: extract_keywords_from_abstract(abstract)["Spacy_noun_chunks"],
    "YAKE_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["YAKE_keywords"],
    "RAKE_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["RAKE_keywords"],
    "Pyate_combo_basic_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["Pyate_combo_basic_keywords"],
    "Pyate_basic_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["Pyate_basic_keywords"],
    "Pyate_cvalues_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["Pyate_cvalues_keywords"],
    "Summa_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["Summa_keywords"],
    "TFIDF_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["TFIDF_keywords"],
    "LSA_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["LSA_keywords"],
    "LDA_keywords": lambda abstract: extract_keywords_from_abstract(abstract)["LDA_keywords"],
}

# %%
base_path = 'data_cs'
datasets = ['Inspec', 'SemEval2010', 'www']

output_folder = 'output'
evaluate_keywords_from_data(base_path, datasets, extraction_functions, output_folder)

# %%
