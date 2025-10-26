import numpy as np

"""
2000 samples
each sample has 7 WIFI signal strengths, last column is room no.
user is standing at. (i.e sample label)
"""

def decision_tree_learning(dataset, depth=0):

    features = dataset[:, :-1]
    label = dataset[:, -1] 

    if (np.all(label == label[0])):
        leaf_node = {
            'leaf' : True,
            'value' : label[0]
        }
        return (leaf_node, depth)
    
    split_feature, threshold = find_split(dataset)

    left = features[:, split_feature] < threshold
    right = ~left

    l_dataset = dataset[left]
    r_dataset = dataset[right]

    left_branch, left_depth = decision_tree_learning(l_dataset, depth+1)
    right_branch, right_depth = decision_tree_learning(r_dataset, depth+1)

    node = {
        'leaf' : False,
        'attribute' : split_feature,
        'value' : threshold,
        'left' : left_branch,
        'right' : right_branch
    }

    return (node, max(left_depth, right_depth))

def find_split(dataset):
    features = dataset[:, :-1]
    label = dataset[:, -1] 

    best_feat = None
    best_th = None
    best_gain = -np.inf

    for feat in range(features.shape[1]):
        thresholds = np.unique(features[:, feat])

        for th in thresholds:
            gain = calc_information_gain(features, label, feat, th)

            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_th = th

    return best_feat, best_th

def calc_information_gain(features, label, feat, threshold):
    left = features[:, feat] < threshold
    right = ~left

    if(np.sum(left) == 0 or np.sum(right) ==    0):
        return 0
    
    H_all = calculate_entropy(label)
    n_total = len(label)
    n_left = np.sum(left)
    n_right = np.sum(right)

    H_left = calculate_entropy(label[left])
    H_right = calculate_entropy(label[right])

    remainder = (n_left / n_total) * H_left + (n_right / n_total) * H_right

    return H_all - remainder

def calculate_entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)

    # 1e-10 to prevent log(0) from being undefined
    H_dataset = -np.sum(p * np.log2(p + 1e-10))

    return H_dataset
    
def print_tree(node, depth=0, prefix="Root: "):

    indent = "  " * depth
    
    if node['leaf']:
        print(f"{indent}{prefix}Leaf (value={node['value']})")
    else:
        print(f"{indent}{prefix}features[{node['attribute']}] < {node['value']}")
        print_tree(node['left'], depth + 1, "L: ")
        print_tree(node['right'], depth + 1, "R: ")
    
def create_confusion_matrix(y_true, y_pred):
    cf_matrix = np.zeros((4,4), dtype=int)

    k = len(cf_matrix)
    class_idx_mapping = {cls : idx for idx, cls in enumerate(range(1,k+1))}
    for true, pred in zip(y_true, y_pred):
        true_idx = class_idx_mapping[true]
        pred_idx = class_idx_mapping[pred]
        cf_matrix[true_idx, pred_idx] += 1

    return cf_matrix

def calc_accuracy(matrix):
    result = np.trace(matrix) / np.sum(matrix)
    return result

def calc_metric(matrix):
    k = len(matrix)
    precision = np.zeros(k)
    recall = np.zeros(k)
    f1_score = np.zeros(k)

    for i in range(k):
        t_p = matrix[i,i]                   # true positives
        f_p = np.sum(matrix[:, i]) - t_p    # false positives
        f_n = np.sum(matrix[i, :]) - t_p    # false negatives

        if (t_p + f_p) > 0:
            precision[i] = t_p / (t_p + f_p)
        else:
            precision[i] = 0

        if (t_p + f_n) > 0:
            recall[i] = t_p / (t_p + f_n)
        else:
            recall[i] = 0

        if (precision[i]+ recall[i]) > 0:
            f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1_score[i] = 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    
def predict_one(sample, tree):
    node = tree
    while not node.get('leaf', False):
        if sample[node['attribute']] < node['value']:
            node = node['left']
        else:
            node = node['right']
    return int(node['value'])

def predict(test_list, tree):
    out = np.empty(test_list.shape[0], dtype=int)
    for i in range(test_list.shape[0]):
        out[i] = predict_one(test_list[i], tree)
    return out        


def kfold_split(dataset, k = 10, seed = 42):     #Break up the dataset into 10 sets

    randomizer = np.random.default_rng(seed)
    label = dataset[:, -1].astype(int)
    classes = np.unique(label)
    # group by class
    class_idxs = []
    for c in classes: 
        idx = np.where(label == c)[0]
        if idx.size > 0:
            perm = randomizer.permutation(idx.size)
            class_idxs.append(idx[perm])
        else:
            class_idxs.append(np.array([], dtype=int))

    folds = [[] for _ in range(k)]
    for c_idx in class_idxs:
        for i, gidx in enumerate(c_idx):
            folds[i % k].append(int(gidx))

    folds = [np.array(sorted(f), dtype=int) for f in folds]
    return folds

def evaluate(test_db, trained_tree):
    test_samples = test_db[:, :-1]
    class_true = test_db[:, -1].astype(int)
    class_pred = predict(test_samples, trained_tree)
    fold_cf = create_confusion_matrix(class_true, class_pred)
    return fold_cf

def cross_validate(dataset, k=10, seed=42, max_depth=None, min_samples_split=2):
    folds = kfold_split(dataset, k=k, seed=seed)
    cf_matrix = np.zeros((4, 4), dtype=int)
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i]) if k > 1 else folds[i]
        train_db = dataset[train_idx]
        test_db = dataset[test_idx]

        tree, _ = decision_tree_learning(train_db, 0)
        fold_cf = evaluate(test_db, tree)
        cf_matrix += fold_cf
    avg_acc = calc_accuracy(cf_matrix)
    test_metric = calc_metric(cf_matrix)
    return cf_matrix, avg_acc, test_metric

if __name__ == "__main__":
    data_clean = np.loadtxt('wifi_db/clean_dataset.txt')
    data_noisy = np.loadtxt('wifi_db/noisy_dataset.txt')
    tree, max_depth = decision_tree_learning(data_clean)
    print_tree(tree)    
    
    
    print("Clean data cross-validation")
    clean_cf_matrix, clean_fold_acc, clean_eval_metric = cross_validate(data_clean,k=10)
    print("Confusion matrix\n", clean_cf_matrix)
    print("Average accuracy",clean_fold_acc)
    print("Class Precision", np.round(clean_eval_metric['precision'],4))
    print("Class Recall", np.round(clean_eval_metric['recall'],4))
    print("ClassF1", np.round(clean_eval_metric['f1_score'],4))
    
    
    print("Noisy data cross-validation")
    noisy_cf_matrix, noisy_fold_acc, noisy_eval_metric = cross_validate(data_noisy,k=10)
    print("Confusion matrix", noisy_cf_matrix)
    print("Average accuracy",noisy_fold_acc)
    print("Class Precision", np.round(noisy_eval_metric['precision'],4))
    print("Class Recall", np.round(noisy_eval_metric['recall'],4))
    print("ClassF1", np.round(noisy_eval_metric['f1_score'],4))


