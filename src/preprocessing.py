'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''
import os
import pandas as pd
import ast

def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    # Your code here
    # Set up the paths to where the CSV files are located
    data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
    model_pred_path = os.path.join(data_folder, 'prediction_model_03.csv')
    genres_path = os.path.join(data_folder, 'genres.csv')

    # Read the prediction model data from the CSV into a DataFrame
    model_pred_df = pd.read_csv(model_pred_path)
    
    # Read the genre information from the CSV into a DataFrame
    genres_df = pd.read_csv(genres_path)
    
    # Let's print out the column names to make sure we loaded everything correctly
    #print("Columns in model_pred_df:", model_pred_df.columns)
    #print("Columns in genres_df:", genres_df.columns)
    
    return model_pred_df, genres_df


def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''

    # Your code here
    # First, we grab a list of all the unique genres from the genres DataFrame
    genre_list = genres_df['genre'].unique().tolist()

    # Now, we set up dictionaries to keep track of how many times each genre appears
    genre_true_counts = {genre: 0 for genre in genre_list}
    genre_tp_counts = {genre: 0 for genre in genre_list}
    genre_fp_counts = {genre: 0 for genre in genre_list}

    # For each row in the predictions DataFrame, we'll count the true, false positive, and true positive genres
    for _, row in model_pred_df.iterrows():
        # Convert the string of genres into a list we can work with
        true_genres = ast.literal_eval(row['actual genres'])
        predicted_genre = row['predicted']

        # Go through each genre in the true genres list
        for true_genre in true_genres:
            true_genre = true_genre.strip()  # Just in case, we remove any extra spaces around the genre name

            if true_genre in genre_true_counts:
                genre_true_counts[true_genre] += 1  # Count how many times this genre actually occurs

            if predicted_genre == true_genre:
                genre_tp_counts[predicted_genre] += 1  # If the prediction matches the true genre, it's a true positive
            else:
                genre_fp_counts[predicted_genre] += 1  # Otherwise, it's a false positive for this predicted genre
    
    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts