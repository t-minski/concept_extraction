import logging
import statistics

# Set up the logger
logger = logging.getLogger(__name__)

def log_statistics(documents, extractive_keyphrases):
    filtered_documents = []
    filtered_keyphrases = []
    unfiltered_keyphrases = []

    keyphrases_per_document = []
    filtered_keyphrases_per_document = []
    unfiltered_keyphrases_per_document = []

    total_documents = len(documents)
    total_keyphrases = 0
    total_filtered_keyphrases = 0
    total_unfiltered_keyphrases = 0

    for tokens, keyphrases in zip(documents, extractive_keyphrases):
        document_text = ' '.join(tokens)
        filtered_phrases = [phrase for phrase in keyphrases if phrase in document_text]
        unfiltered_phrases = [phrase for phrase in keyphrases if phrase not in document_text]
        
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

    # Log statistics
    logger.info(f"Total documents: {total_documents}")
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

    return filtered_documents, filtered_keyphrases
