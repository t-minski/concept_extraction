import logging

# Set up the logger
logger = logging.getLogger(__name__)

def log_statistics(documents, extractive_keyphrases):
    logger.info(f"Filtering documents with no keyphrases...")
    filtered_documents = []
    filtered_keyphrases = []
    total_documents_with_no_keyphrases = 0
    
    for tokens, keyphrases in zip(documents, extractive_keyphrases):        
        document_text = ' '.join(tokens)
        filtered_phrases = [phrase for phrase in keyphrases if phrase in document_text]

        if not filtered_phrases:  # Skip if there are no ground truth keyphrases
            total_documents_with_no_keyphrases += 1
            continue

        filtered_documents.append(document_text)
        filtered_keyphrases.append(filtered_phrases)

    # Log statistics
    logger.info(f"Documents with no keyphrases: {total_documents_with_no_keyphrases} / {len(documents)} filtered")

    return filtered_documents, filtered_keyphrases
