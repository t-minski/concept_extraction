import pytest
from conexion.evaluation.evaluator import evaluate_p_r_f_at_k

def test_ndcg_group_x():

    keyphrases = [
        ('item_d', 0.51),
        ('item_e', 0.01),
        ('item_a', 0.95),        
        ('item_c', 0.88),
        ('item_b', 0.90),
    ]
    # https://towardsdatascience.com/demystifying-ndcg-bee3be58cfe0  search group x
    references = ['item_c', 'item_d', 'item_e']


    prf_5, prf_10, prf_15, ndcg, map =  evaluate_p_r_f_at_k(keyphrases, references)
    assert(ndcg == pytest.approx(0.61828, abs=0.00001))


def test_ndcg_group_y():

    keyphrases = [
        ('item_j', 0.50),
        ('item_h', 0.70),
        ('item_f', 0.90),
        ('item_i', 0.60),
        ('item_g', 0.80),
    ]
    # https://towardsdatascience.com/demystifying-ndcg-bee3be58cfe0  search group y
    references = ['item_f', 'item_h', 'item_j']

    prf_5, prf_10, prf_15, ndcg, map =  evaluate_p_r_f_at_k(keyphrases, references)
    assert(ndcg == pytest.approx(0.88546, abs=0.00001))

def test_mean_average_precision():
    # https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
    # 2. Average Precision and mAP for Information Retrieval
    keyphrases = [
        ('pos_three', 0.80),
        ('pos_one', 0.99),
        ('pos_six', 0.50),
        ('pos_two', 0.90),
        ('pos_four', 0.70),
        ('pos_five', 0.60),
        
    ]
    # https://towardsdatascience.com/demystifying-ndcg-bee3be58cfe0  search group y
    references = ['pos_one', 'pos_four', 'pos_five']

    prf_5, prf_10, prf_15, ndcg, map =  evaluate_p_r_f_at_k(keyphrases, references)
    assert(map == pytest.approx(0.7))