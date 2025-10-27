import numpy as np
import matplotlib.pyplot as plt

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

def create_confusion_matrix(y_true, y_pred, cf_matrix):
    
    k = len(cf_matrix)

    class_idx_mapping = {cls : idx for idx, cls in enumerate(k)}
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

def visualise_tree(node, spacing=3.0, dy=2.0, cmap='tab20'):
    pos = {}
    _set_pos(node, 0, pos, counter=[0], spacing=spacing)
    max_depth = max(y for _, y in pos.values())

    fig, ax = plt.subplots(figsize=(10, (max_depth + 1) * dy))
    ax.axis("off")
    _draw(ax, node, pos, 0, max_depth, dy, plt.cm.get_cmap(cmap))
    plt.show()

def _set_pos(node, depth, pos, counter, spacing):
    if node["leaf"]:
        x = counter[0] * spacing
        pos[id(node)] = (x, depth)
        counter[0] += 1
        return x
    lx = _set_pos(node["left"], depth + 1, pos, counter, spacing)
    rx = _set_pos(node["right"], depth + 1, pos, counter, spacing)
    pos[id(node)] = ((lx + rx) / 2, depth)
    return (lx + rx) / 2

def _draw(ax, node, pos, depth, max_depth, dy, cmap):
    x, y = pos[id(node)]
    y = -y * dy
    color = cmap(depth / max_depth)

    # Draw node
    label = f"leaf:{node['value']:.3f}" if node["leaf"] else f"[X{node['attribute']}<{node['value']}]"
    ax.text(x, y, label, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#e6f5d0" if not node["leaf"] else "#f0f8ff"),
            fontsize=9)

    # Draw edges
    if not node["leaf"]:
        for i, (child, tag) in enumerate(zip([node["left"], node["right"]], ["T", "F"])):
            cx, cy = pos[id(child)]
            cy = -cy * dy
            ax.plot([x, cx], [y - 0.1, cy + 0.1], color=color, lw=2)
            ax.text((x + cx) / 2, (y + cy) / 2, tag, ha="center", va="center",
                    fontsize=8, fontweight="bold", color="black", alpha=0.8)
            _draw(ax, child, pos, depth + 1, max_depth, dy, cmap)

if __name__ == "__main__":
    data_clean = np.loadtxt('wifi_db/clean_dataset.txt')

    tree, max_depth = decision_tree_learning(data_clean)
    print_tree(tree)
    visualise_tree(tree)

