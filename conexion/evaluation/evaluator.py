def evaluate_p_r_f(keyphrases, references):
    P = len(set(keyphrases) & set(references)) / len(keyphrases)
    R = len(set(keyphrases) & set(references)) / len(references)
    F = (2*P*R)/(P+R) if (P+R) > 0 else 0 
    return (P, R, F)


def evaluate(models, datasets, output_folder):
    # Evaluate the models on the datasets
    for dataset in datasets:
        training_abstracts, training_concepts = dataset.get_training_data()
        test_abstracts, test_concepts = dataset.get_test_data()
        for model in models:
            type_model = type(model)
            # Train the model
            model.fit(training_abstracts, training_concepts)
            
            # Predict the concepts
            predicted_concepts_with_confidence = model.predict(test_abstracts)
            
            # Evaluate the predictions
            all_evaluation_results = []
            cumulative_precision, cumulative_recall, cumulative_f1_score = 0, 0, 0
            assert len(test_abstracts) == len(predicted_concepts_with_confidence)
            
            # Calculate the precision, recall, and F1 score
            for i, abstract in enumerate(test_abstracts):
                only_keyword = [keyword for (keyword, confidence_score) in predicted_concepts_with_confidence[i]]
                precision, recall, f1_score = evaluate_p_r_f(only_keyword, test_concepts[i])

                cumulative_precision += precision
                cumulative_recall += recall
                cumulative_f1_score += f1_score

                all_evaluation_results.append((test_concepts[i], only_keyword, precision, recall, f1_score, len(test_concepts[i]), len(only_keyword)))
        
            average_precision = cumulative_precision / len(test_abstracts)
            average_recall = cumulative_recall / len(test_abstracts)
            average_f1_score =  cumulative_f1_score / len(test_abstracts)

            # Write ground truth keywords, extracted keywords, and evaluation results to CSV files
            with open(os.path.join(output_folder, f'evaluation_results_{type_model}_{dataset}.csv'), 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Ground_truth Keywords', 'Extracted Keywords', 'Precision', 'Recall', 'F1-score', 'n_gt_keywords', 'n_extraced_leywords'])
                writer.writerows(all_evaluation_results)
            
            with open(os.path.join(output_folder, f'evaluation_results_avg_{type_model}_{dataset}.csv'), 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Precision', 'Recall', 'F1-score'])
                writer.writerows((average_precision, average_recall, average_f1_score))

                