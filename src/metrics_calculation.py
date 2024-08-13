'''
PART 2: METRICS CALCULATION
- Tailor the code scaffolding below to calculate various metrics
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import ast

def calculate_metrics(model_pred_df, genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts):
    '''
    Calculate micro and macro metrics
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    
    Returns:
        tuple: Micro precision, recall, F1 score
        lists of macro precision, recall, and F1 scores
    
    Hint #1: 
    tp -> true positives
    fp -> false positives
    tn -> true negatives
    fn -> false negatives

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    Hint #2: Micro metrics are tuples, macro metrics are lists

    '''

    # Your code here
     # Let's start by calculating the total true positives, false positives, and false negatives for micro metrics
    micro_tp = sum(genre_tp_counts.values())
    micro_fp = sum(genre_fp_counts.values())
    micro_fn = sum(genre_true_counts.values()) - micro_tp

    # Calculate micro precision, recall, and F1 score
    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    # Now let's prepare to calculate macro metrics by initializing empty lists
    macro_prec_list = []
    macro_recall_list = []
    macro_f1_list = []

    # We loop through each genre and calculate precision, recall, and F1 score for each one
    for genre in genre_list:
        tp = genre_tp_counts.get(genre, 0)
        fp = genre_fp_counts.get(genre, 0)
        fn = genre_true_counts.get(genre, 0) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        macro_prec_list.append(precision)
        macro_recall_list.append(recall)
        macro_f1_list.append(f1)

    # Finally, we return the micro metrics as a tuple and the macro metrics as lists
    return micro_precision, micro_recall, micro_f1, macro_prec_list, macro_recall_list, macro_f1_list
    
def calculate_sklearn_metrics(model_pred_df, genre_list):
    '''
    Calculate metrics using sklearn's precision_recall_fscore_support.
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions.
        genre_list (list): List of unique genres.
    
    Returns:
        tuple: Macro precision, recall, F1 score, and micro precision, recall, F1 score.
    
    Hint #1: You'll need these two lists
    pred_rows = []
    true_rows = []
    
    Hint #2: And a little later you'll need these two matrixes for sk-learn
    pred_matrix = pd.DataFrame(pred_rows)
    true_matrix = pd.DataFrame(true_rows)
    '''

    # Your code here
    true_rows = []
    pred_rows = []

     # We need to go through each row in the DataFrame and convert the true genres and predicted genres into a format sklearn can use
    for _, row in model_pred_df.iterrows():
        true_genres = ast.literal_eval(row['actual genres'])  # Convert the string of genres into a list
        predicted_genre = row['predicted']
        
        # Create a binary list for true genres (1 if the genre is present, 0 if not)
        true_row = [1 if genre in true_genres else 0 for genre in genre_list]
        # Do the same for the predicted genre
        pred_row = [1 if genre == predicted_genre else 0 for genre in genre_list]
        
        true_rows.append(true_row)
        pred_rows.append(pred_row)

    # Convert the lists into DataFrames so sklearn can process them
    true_matrix = pd.DataFrame(true_rows, columns=genre_list)
    pred_matrix = pd.DataFrame(pred_rows, columns=genre_list)

    # Now let's calculate precision, recall, and F1 score for each genre using sklearn
    precision, recall, f1, _ = precision_recall_fscore_support(true_matrix.values, pred_matrix.values, average=None, zero_division=0)
    
    # Calculate macro metrics by averaging the precision, recall, and F1 scores across all genres
    macro_prec = np.mean(precision)
    macro_rec = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Calculate micro metrics by treating the entire matrix as a single set of predictions
    micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(true_matrix.values, pred_matrix.values, average='micro', zero_division=0)

    # Return the macro and micro metrics as tuples
    return macro_prec, macro_rec, macro_f1, micro_prec, micro_rec, micro_f1