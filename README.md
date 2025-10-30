## Overview
This implementation builds a decision tree classifier to predict indoor room location (1-4) based on WiFi signal strengths from 7 emitters. The algorithm uses information gain with entropy for splitting and supports 10-fold cross-validation evaluation.

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/lucasvenetz/CW1_DecisionTree
2. Install requirements:
   ```bash  
   pip install -r For 60012/requirements.txt

## Running the Code
```bash
   python3 decisionTree.py
```

Main functions:
- decision_tree_learning(dataset, depth) - Recursive tree building
- find_split(dataset) - Find optimal split using information gain
- cross_validate(dataset, k) - Perform k-fold cross-validation
- evaluate(test_db, trained_tree) - Evaluate tree on test data
- visualise_tree(node) - Display graphical tree visualization
- print_tree(node) - Print text representation of tree


## Expected Output
- Two 4x4 confusion matrices for each dataset
- Average accuracy across all folds
- Per-class precision, recall and F1-Score metrics
- Tree Structure (one in ASCII and one using matplotlib) 
