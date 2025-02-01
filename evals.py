"""
main parameters used:

qrels = {query: {relevant_doc_1: score which is 1 or 0 most often, relevant_doc_2: score...}}
results = {query: {retrieved_doc_1: score , retrieved_doc_2: score...}}
k_values = [cutoff values]

metrics to measure:
- recall@k
- precision@k
- MAP@k
- top-k accuracy
- NDCG@k

"""

import pytrec_eval

def top_k_accuracy(qrels, results, k_values):
    top_k_acc = dict()
    
    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = 0.0
    
    k_max, top_hits = max(k_values), {}
    
    for query_id, doc_scores in results.items():
        top_hits[query_id] = [item[0] for item in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]]
    
    for query_id in top_hits:
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])
        for k in k_values:
            for relevant_doc_id in query_relevant_docs:
                if relevant_doc_id in top_hits[query_id][0:k]:
                    top_k_acc[f"Accuracy@{k}"] += 1.0
                    break

    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = round(top_k_acc[f"Accuracy@{k}"]/len(qrels), 5)

    return top_k_acc

def evaluate(qrels, results, k_values):
    recall = dict()
    precision = dict()
    _map = dict()
    ndcg = dict()

    for k in k_values:
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        ndcg[f"NDCG@{k}"] = 0.0

    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
    
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_"+ str(k)]
    
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)

    top_k_accuracy_results = top_k_accuracy(qrels, results, k_values)

    return ndcg, _map, recall, precision, top_k_accuracy_results