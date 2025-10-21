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

if __name__ == "__main__":
    data_clean = np.loadtxt('wifi_db/clean_dataset.txt')

    tree, max_depth = decision_tree_learning(data_clean)
    print_tree(tree)
