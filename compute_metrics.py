'''
Load the output_test.txt file and compute the metrics
'''


import random
from collections import defaultdict
import metrics


test_out_filename = "./output/Ubuntu_V1_Xu/1596330255/output_test.txt"  # modify this variable to the path to the testing model
print("*"*20 + test_out_filename + "*"*20 + "\n")

with open(test_out_filename, 'r') as f:
    
	# candidate size = 10
    results = defaultdict(list)
    lines = f.readlines()
    for line in lines[1:]:
        line = line.strip().split('\t')
        us_id = line[0]
        r_id = line[1]
        prob_score = float(line[2])
        label = float(line[4])
        results[us_id].append((r_id, label, prob_score))

    accu, precision, recall, f1, loss = metrics.classification_metrics(results)
    print('Accuracy: {}, Precision: {}  Recall: {}  F1: {} Loss: {}'.format(accu, precision, recall, f1, loss))
    total_valid_query = metrics.get_num_valid_query(results)
    mvp = metrics.mean_average_precision(results)
    mrr = metrics.mean_reciprocal_rank(results)
    print('MAP (mean average precision: {}\tMRR (mean reciprocal rank): {}\tNum_query: {}'.format(
        mvp, mrr, total_valid_query))
    top_1_precision = metrics.top_k_precision(results, k=1)
    top_2_precision = metrics.top_k_precision(results, k=2)
    top_5_precision = metrics.top_k_precision(results, k=5)
    print('Recall_10@1: {}\tRecall_10@2: {}\tRecall_10@5: {}\n'.format(
        top_1_precision, top_2_precision, top_5_precision))

    # candidate size = 2, the results may vary at different runs because we sample the negative candidate randomly
    results_bin = defaultdict(list)
    for us_id, candidates in results.items():
    	false_candidates = []
    	for candidate in candidates:
    		r_id, label, prob_score = candidate
    		if label == 1.0:
    			results_bin[us_id].append(candidate)
    		if label == 0.0:
    			false_candidates.append(candidate)
    	false_candidate = random.sample(false_candidates, 1)
    	results_bin[us_id].append(false_candidate[0])

    accu, precision, recall, f1, loss = metrics.classification_metrics(results_bin)
    print('Accuracy: {}, Precision: {}  Recall: {}  F1: {} Loss: {}'.format(accu, precision, recall, f1, loss))
    total_valid_query = metrics.get_num_valid_query(results_bin)
    mvp = metrics.mean_average_precision(results_bin)
    mrr = metrics.mean_reciprocal_rank(results_bin)
    top_1_precision = metrics.top_k_precision(results_bin, k=1)
    print('MAP (mean average precision: {}\tMRR (mean reciprocal rank): {}\tNum_query: {}'.format(
        mvp, mrr, total_valid_query))
    print('Recall_2@1: {}\n'.format(
        top_1_precision))
