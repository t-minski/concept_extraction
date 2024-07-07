import argparse
from typing import List
from conexion.data import get_datasets
import spacy
import logging
import statistics

logger = logging.getLogger(__name__)

# Load models
nlp_sci = spacy.load("en_core_sci_lg")
nlp_web = spacy.load("en_core_web_lg")

def analyze_datasets(datasets: List[str]):
    
    for dataset in get_datasets(datasets):
        training_abstracts, training_concepts = dataset.get_training_data()
        test_abstracts, test_concepts = dataset.get_test_data()



        filtered_documents = []
        filtered_keyphrases = []
        unfiltered_keyphrases = []

        keyphrases_per_document = []
        filtered_keyphrases_per_document = []
        unfiltered_keyphrases_per_document = []

        total_documents = len(documents)
        total_documents_with_no_keyphrases = 0
        total_keyphrases = 0
        total_filtered_keyphrases = 0
        total_unfiltered_keyphrases = 0

        total_named_entity_keyphrases = 0
        total_noun_phrase_keyphrases = 0
        
        for tokens, keyphrases in zip(documents, extractive_keyphrases):        
            document_text = ' '.join(tokens)
            filtered_phrases = [phrase for phrase in keyphrases if phrase in document_text]
            unfiltered_phrases = [phrase for phrase in keyphrases if phrase not in document_text]
            
            if not filtered_phrases:  # Skip if there are no ground truth keyphrases
                total_documents_with_no_keyphrases += 1
                continue
            
            # Named entity recognition
            doc_sci = nlp_sci(document_text)
            named_entities = {ent.text for ent in doc_sci.ents}
            named_entity_keyphrases = [phrase for phrase in filtered_phrases if phrase in named_entities]
            total_named_entity_keyphrases += len(named_entity_keyphrases)
            
            # Noun phrase detection
            doc_web = nlp_web(document_text)
            noun_phrases = {chunk.text for chunk in doc_web.noun_chunks if not chunk.root.dep_ == 'det'}
            noun_phrase_keyphrases = [phrase for phrase in filtered_phrases if phrase in noun_phrases]
            total_noun_phrase_keyphrases += len(noun_phrase_keyphrases)

            filtered_documents.append(document_text)
            filtered_keyphrases.append(filtered_phrases)
            unfiltered_keyphrases.extend(unfiltered_phrases)
            
            keyphrases_per_document.append(len(keyphrases))
            filtered_keyphrases_per_document.append(len(filtered_phrases))
            unfiltered_keyphrases_per_document.append(len(unfiltered_phrases))
            
            total_keyphrases += len(keyphrases)
            total_filtered_keyphrases += len(filtered_phrases)
            total_unfiltered_keyphrases += len(unfiltered_phrases)

        # Calculate averages and other statistics
        average_keyphrases_per_document = statistics.mean(keyphrases_per_document)
        average_filtered_keyphrases_per_document = statistics.mean(filtered_keyphrases_per_document)
        average_unfiltered_keyphrases_per_document = statistics.mean(unfiltered_keyphrases_per_document)

        max_keyphrases = max(keyphrases_per_document)
        min_keyphrases = min(keyphrases_per_document)
        stddev_keyphrases = statistics.stdev(keyphrases_per_document)

        max_filtered_keyphrases = max(filtered_keyphrases_per_document)
        min_filtered_keyphrases = min(filtered_keyphrases_per_document)
        stddev_filtered_keyphrases = statistics.stdev(filtered_keyphrases_per_document)

        max_unfiltered_keyphrases = max(unfiltered_keyphrases_per_document)
        min_unfiltered_keyphrases = min(unfiltered_keyphrases_per_document)
        stddev_unfiltered_keyphrases = statistics.stdev(unfiltered_keyphrases_per_document)

        # Calculate percentages
        percentage_named_entity_keyphrases = (total_named_entity_keyphrases / total_filtered_keyphrases) * 100 if total_filtered_keyphrases else 0
        percentage_noun_phrase_keyphrases = (total_noun_phrase_keyphrases / total_filtered_keyphrases) * 100 if total_filtered_keyphrases else 0

        # Log statistics
        logger.info(f"Total documents: {total_documents}")
        logger.info(f"Total documents with no keyphrases: {total_documents_with_no_keyphrases}")
        logger.info(f"Total keyphrases: {total_keyphrases}")
        logger.info(f"Total filtered keyphrases: {total_filtered_keyphrases}")
        logger.info(f"Total unfiltered keyphrases: {total_unfiltered_keyphrases}")

        logger.info(f"Average keyphrases per document: {average_keyphrases_per_document}")
        logger.info(f"Max keyphrases in a document: {max_keyphrases}")
        logger.info(f"Min keyphrases in a document: {min_keyphrases}")
        logger.info(f"Standard deviation of keyphrases per document: {stddev_keyphrases}")

        logger.info(f"Average filtered keyphrases per document: {average_filtered_keyphrases_per_document}")
        logger.info(f"Max filtered keyphrases in a document: {max_filtered_keyphrases}")
        logger.info(f"Min filtered keyphrases in a document: {min_filtered_keyphrases}")
        logger.info(f"Standard deviation of filtered keyphrases per document: {stddev_filtered_keyphrases}")

        logger.info(f"Average unfiltered keyphrases per document: {average_unfiltered_keyphrases_per_document}")
        logger.info(f"Max unfiltered keyphrases in a document: {max_unfiltered_keyphrases}")
        logger.info(f"Min unfiltered keyphrases in a document: {min_unfiltered_keyphrases}")
        logger.info(f"Standard deviation of unfiltered keyphrases per document: {stddev_unfiltered_keyphrases}")

        logger.info(f"Percentage of keyphrases that are named entities: {percentage_named_entity_keyphrases:.2f}%")
        logger.info(f"Percentage of keyphrases that are noun phrases: {percentage_noun_phrase_keyphrases:.2f}%")


        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets", "-d", nargs='+', help="Name of datasets e.g. `inspec`"
    )
    args = parser.parse_args()
    
    analyze_datasets(args.datasets)