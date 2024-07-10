import os
import csv
from typing import List, Tuple
import logging
import time
from sklearn.metrics import ndcg_score, average_precision_score
from conexion.data.base_dataset import BaseDataset
from conexion.models.base_model import BaseModel
from nltk.stem import PorterStemmer
import spacy

nlp = spacy.load("en_core_web_sm")
logger = logging.getLogger(__name__)



def evaluate_p_r_f(keyphrases: List[str], references : List[str]):
    P = len(set(keyphrases) & set(references)) / len(keyphrases) if len(keyphrases) > 0 else 0
    R = len(set(keyphrases) & set(references)) / len(references) if len(references) > 0 else 0
    F = (2*P*R)/(P+R) if (P+R) > 0.0 else 0.0
    return (P, R, F)

def evaluate_p_r_f_at_k(keyphrases : List[Tuple[str, float]], references : List[str]):
    #sort the keypharases based on the confidence score
    keyphrases.sort(key=lambda x: x[1], reverse=True)
    #logger.debug(f"keyphrases (sorted): {keyphrases}")
    values = []
    for k in [5, 10, 15]:
        top_k_keyphrase_set = set([keyphrase for keyphrase, conf_value in keyphrases[:k]])
        correct_predictions = len(top_k_keyphrase_set & set(references))
        P = correct_predictions / len(top_k_keyphrase_set) if len(top_k_keyphrase_set) > 0 else 0 # TODO: horrible metric -> check cases where ground truth has less than k keyphrases
        R = correct_predictions / len(references) if len(references) > 0 else 0 # TODO: horrible metric: system cannot get to 1.0 recall because of the top k keyphrases
        F = (2*P*R)/(P+R) if (P+R) > 0.0 else 0.0
        values.append((P, R, F))
    
    # combine the keyphrases and the references
    all_keywords = set(references).union([keyphrase for keyphrase, conf_value in keyphrases])
    y_true = [1 if keyphrase in references else 0 for keyphrase in all_keywords]        
    y_scores = []
    for keyphrase in all_keywords:
        for keyphrase_pred, conf_value in keyphrases:
            if keyphrase == keyphrase_pred:
                y_scores.append(conf_value)
                break
        else:
            y_scores.append(0.0)
    
    if len(all_keywords) == 1:
        values.append(1.0 if len(set(references).intersection([keyphrase for keyphrase, conf_value in keyphrases])) > 0 else 0.0)
    else:
        ndcg = ndcg_score([y_true], [y_scores])
        values.append(ndcg)

        
    average_precision = average_precision_score(y_true, y_scores)
    values.append(average_precision)

    return values

def evaluate(models: List[BaseModel], datasets: List[BaseDataset], output_folder: str):
    evaluate_transfer_learning(models, [(dataset, dataset) for dataset in datasets], output_folder)

def evaluate_transfer_learning(models : List[BaseModel], datasets : List[Tuple[BaseDataset, BaseDataset]], output_folder : str):
    # Evaluate the models on the datasets
    for training_dataset, test_dataset in datasets:
        type_training_dataset = training_dataset.__class__.__name__
        type_test_dataset = test_dataset.__class__.__name__
        type_dataset = f"{type_training_dataset}-{type_test_dataset}"
        
        training_abstracts, training_concepts = training_dataset.get_training_data()
        test_abstracts, test_concepts = test_dataset.get_test_data()
        for model in models:
            type_model = getattr(model, "model_template_name", model.__class__.__name__)

            
            # Train the model
            logger.info(f"Run training of model {type_model} on {type_dataset}")
            start_time = time.time()
            model.fit(training_abstracts, training_concepts)
            training_time = time.time() - start_time

            # Predict the concepts
            logger.info(f"Run prediction of model {type_model} on {type_dataset}")
            start_time = time.time()
            predicted_concepts_with_confidence = model.predict(test_abstracts)
            prediction_time = time.time() - start_time
            
            logger.info(f"Run evaluation of model {type_model} on {type_dataset}")
            # Evaluate the predictions
            cumulative_precision, cumulative_recall, cumulative_f1_score = 0, 0, 0
            stemmed_cumulative_precision, stemmed_cumulative_recall, stemmed_cumulative_f1_score = 0, 0, 0
            
            cumulative_gt_keywords = 0
            cumulative_extracted_keywords = 0
            
            cumulative_precision_5, cumulative_recall_5, cumulative_f1_score_5 = 0, 0, 0
            stemmed_cumulative_precision_5, stemmed_cumulative_recall_5, stemmed_cumulative_f1_score_5 = 0, 0, 0
            
            cumulative_precision_10, cumulative_recall_10, cumulative_f1_score_10 = 0, 0, 0
            stemmed_cumulative_precision_10, stemmed_cumulative_recall_10, stemmed_cumulative_f1_score_10 = 0, 0, 0
            
            cumulative_precision_15, cumulative_recall_15, cumulative_f1_score_15 = 0, 0, 0
            stemmed_cumulative_precision_15, stemmed_cumulative_recall_15, stemmed_cumulative_f1_score_15 = 0, 0, 0

            assert len(test_abstracts) == len(predicted_concepts_with_confidence)
            
            with open(os.path.join(output_folder, f'evaluation_results-{type_model}-{type_dataset}.csv'), 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Ground_truth Keywords', 'Extracted Keywords', 'Precision', 'Recall', 'F1-score', 'n_gt_keywords', 'n_extraced_leywords',
                                 'P@5', 'R@5', 'F@5', 'P@10', 'R@10', 'F@10', 'P@15', 'R@15', 'F@15', 'NDCG', 'MAP',
                                 'stemmed_Precision', 'stemmed_Recall', 'stemmed_F1-score', 'stemmed_P@5', 'stemmed_R@5', 'stemmed_F@5', 'stemmed_P@10', 'stemmed_R@10', 'stemmed_F@10', 'stemmed_P@15', 'stemmed_R@15', 'stemmed_F@15', 'stemmed_NDCG', 'stemmed_MAP'
                                 ])
                # Calculate the precision, recall, and F1 score
                for i, abstract in enumerate(test_abstracts):
                    only_keyword = [keyword for (keyword, confidence_score) in predicted_concepts_with_confidence[i]]
                    precision, recall, f1_score = evaluate_p_r_f(only_keyword, test_concepts[i])
                    prf_5, prf_10, prf_15, ndcg, map = evaluate_p_r_f_at_k(predicted_concepts_with_confidence[i], test_concepts[i])
                    
                    # tokenization and stemming
                    stemmed_only_keyword = [" ".join([PorterStemmer().stem(tok.text.lower()) for tok in nlp(keyphrase)]) for keyphrase in only_keyword ]
                    stemmed_test = [" ".join([PorterStemmer().stem(tok.text.lower()) for tok in nlp(keyphrase)]) for keyphrase in test_concepts[i] ]
                    stemmed_predicted_concepts_with_confidence = [" ".join([PorterStemmer().stem(tok.text.lower()) for tok in nlp(keyphrase)]) for keyphrase, confidence in predicted_concepts_with_confidence[i] ]
                    stemmed_predicted_concepts_with_confidence = [(" ".join([PorterStemmer().stem(tok.text.lower()) for tok in nlp(keyphrase)]), confidence) for keyphrase, confidence in predicted_concepts_with_confidence[i]]

                    stemmed_precision, stemmed_recall, stemmed_f1_score = evaluate_p_r_f(stemmed_only_keyword, stemmed_test)
                    stemmed_prf_5, stemmed_prf_10, stemmed_prf_15, stemmed_ndcg, stemmed_map = evaluate_p_r_f_at_k(stemmed_predicted_concepts_with_confidence, stemmed_test)
                    
                    stemmed_cumulative_precision += stemmed_precision
                    stemmed_cumulative_recall += stemmed_recall
                    stemmed_cumulative_f1_score += stemmed_f1_score

                    stemmed_cumulative_precision_5 += stemmed_prf_5[0]
                    stemmed_cumulative_recall_5 += stemmed_prf_5[1]
                    stemmed_cumulative_f1_score_5 += stemmed_prf_5[2]

                    stemmed_cumulative_precision_10 += stemmed_prf_10[0]
                    stemmed_cumulative_recall_10 += stemmed_prf_10[1]
                    stemmed_cumulative_f1_score_10 += stemmed_prf_10[2]

                    stemmed_cumulative_precision_15 += stemmed_prf_15[0]
                    stemmed_cumulative_recall_15 += stemmed_prf_15[1]
                    stemmed_cumulative_f1_score_15 += stemmed_prf_15[2]
                    
                    # not stemmed
                    cumulative_precision += precision
                    cumulative_recall += recall
                    cumulative_f1_score += f1_score

                    cumulative_precision_5 += prf_5[0]
                    cumulative_recall_5 += prf_5[1]
                    cumulative_f1_score_5 += prf_5[2]

                    cumulative_precision_10 += prf_10[0]
                    cumulative_recall_10 += prf_10[1]
                    cumulative_f1_score_10 += prf_10[2]

                    cumulative_precision_15 += prf_15[0]
                    cumulative_recall_15 += prf_15[1]
                    cumulative_f1_score_15 += prf_15[2]

                    cumulative_gt_keywords += len(test_concepts[i])
                    cumulative_extracted_keywords += len(only_keyword)

                    writer.writerow((test_concepts[i], only_keyword, precision, recall, f1_score, len(test_concepts[i]), len(only_keyword),
                                     *prf_5, *prf_10, *prf_15, ndcg, map, stemmed_precision, stemmed_recall, stemmed_f1_score, *stemmed_prf_5, *stemmed_prf_10, *stemmed_prf_15, stemmed_ndcg, stemmed_map))
                    

            stemmed_average_precision = stemmed_cumulative_precision / len(test_abstracts)
            stemmed_average_recall = stemmed_cumulative_recall / len(test_abstracts)
            stemmed_average_f1_score =  stemmed_cumulative_f1_score / len(test_abstracts)

            stemmed_average_precision_5 = stemmed_cumulative_precision_5 / len(test_abstracts)
            stemmed_average_recall_5 = stemmed_cumulative_recall_5 / len(test_abstracts)
            stemmed_average_f1_score_5 =  stemmed_cumulative_f1_score_5 / len(test_abstracts)

            stemmed_average_precision_10 = stemmed_cumulative_precision_10 / len(test_abstracts)
            stemmed_average_recall_10 = stemmed_cumulative_recall_10 / len(test_abstracts)
            stemmed_average_f1_score_10 =  stemmed_cumulative_f1_score_10 / len(test_abstracts)

            stemmed_average_precision_15 = stemmed_cumulative_precision_15 / len(test_abstracts)
            stemmed_average_recall_15 = stemmed_cumulative_recall_15 / len(test_abstracts)
            stemmed_average_f1_score_15 =  stemmed_cumulative_f1_score_15 / len(test_abstracts)

            
            # not stemmed
            average_precision = cumulative_precision / len(test_abstracts)
            average_recall = cumulative_recall / len(test_abstracts)
            average_f1_score =  cumulative_f1_score / len(test_abstracts)

            average_precision_5 = cumulative_precision_5 / len(test_abstracts)
            average_recall_5 = cumulative_recall_5 / len(test_abstracts)
            average_f1_score_5 =  cumulative_f1_score_5 / len(test_abstracts)

            average_precision_10 = cumulative_precision_10 / len(test_abstracts)
            average_recall_10 = cumulative_recall_10 / len(test_abstracts)
            average_f1_score_10 =  cumulative_f1_score_10 / len(test_abstracts)

            average_precision_15 = cumulative_precision_15 / len(test_abstracts)
            average_recall_15 = cumulative_recall_15 / len(test_abstracts)
            average_f1_score_15 =  cumulative_f1_score_15 / len(test_abstracts)

            average_gt_keywords = cumulative_gt_keywords / len(test_abstracts)
            average_extracted_keywords = cumulative_extracted_keywords / len(test_abstracts)

            # Write ground truth keywords, extracted keywords, and evaluation results to CSV files
            
            with open(os.path.join(output_folder, f'evaluation_results_avg-{type_model}-{type_dataset}.csv'), 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Precision', 'Recall', 'F1-score', 'avg_n_gt_keywords', 'avg_n_extraced_leywords',
                                 'P@5', 'R@5', 'F@5', 'P@10', 'R@10', 'F@10', 'P@15', 'R@15', 'F@15', 'NDCG', 'MAP',
                                 'stemmed_Precision', 'stemmed_Recall', 'stemmed_F1-score',
                                 'stemmed_P@5', 'stemmed_R@5', 'stemmed_F@5', 'stemmed_P@10', 'stemmed_R@10', 'stemmed_F@10', 'stemmed_P@15', 'stemmed_R@15', 'stemmed_F@15', 'stemmed_NDCG', 'stemmed_MAP',
                                 'training seconds', 'prediction seconds'])
                writer.writerow([average_precision, average_recall, average_f1_score, average_gt_keywords, average_extracted_keywords,
                                 average_precision_5, average_recall_5, average_f1_score_5,
                                 average_precision_10, average_recall_10, average_f1_score_10,
                                 average_precision_15, average_recall_15, average_f1_score_15,
                                 ndcg, map,
                                 stemmed_average_precision, stemmed_average_recall, stemmed_average_f1_score, stemmed_average_precision_5, stemmed_average_recall_5, stemmed_average_f1_score_5,
                                 stemmed_average_precision_10, stemmed_average_recall_10, stemmed_average_f1_score_10, stemmed_average_precision_15, stemmed_average_recall_15, stemmed_average_f1_score_15,
                                 stemmed_ndcg, stemmed_map,
                                 training_time, prediction_time])
            
            logger.info(f"Finished evaluation of model {type_model} on {type_dataset}")
