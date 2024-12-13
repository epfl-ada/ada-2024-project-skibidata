import time
import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SPARQLWrapper import SPARQLWrapper, JSON
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score


def extract_from_dict(string):
    if isinstance(string, str) and string != '{}':
        return ', '.join(re.findall(r'": "([^"]+)"', string))
    return np.nan


def wikipedia_query(save_file=True, save_path='', filename='wiki.csv'):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    sparql.setQuery("""
    SELECT ?work ?imdb_id ?freebase_id 
    WHERE
    {
      ?work wdt:P31/wdt:P279* wd:Q11424.
      ?work wdt:P345 ?imdb_id.
      ?work wdt:P646 ?freebase_id.

      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    }
    """)

    sparql.setReturnFormat(JSON)

    try:
        # Execute the query and convert results to a DataFrame
        results = sparql.query().convert()
        print('Wikipedia query successful')
        bindings = results['results']['bindings']

        # Extracting IMDb, Freebase IDs
        data = []
        for binding in bindings:
            row = {
                'imdb_id': binding['imdb_id']['value'] if 'imdb_id' in binding else np.nan,
                'freebase_ID': binding['freebase_id']['value'] if 'freebase_id' in binding else np.nan,
            }
            data.append(row)

        wiki_df = pd.DataFrame(data)
        wiki_df_filtered = wiki_df.drop_duplicates('imdb_id', keep='first')

        if save_file:
            wiki_df_filtered.to_csv(save_path + filename, index=False)
            print(f'file {save_path + filename} saved')

        return wiki_df_filtered

    except Exception as e:
        print(f"An error occurred: {e}")


def show_missing_values(df):
    missing_counts = df.isna().sum()
    missing_proportions = df.isna().mean()

    missing_info = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing Proportion': missing_proportions
    })
    return missing_info


def merge_tmdb_datasets(df_tmdb_movie, df_tmdb_keywords, df_tmdb_credits):

    df_movie = df_tmdb_movie.copy()
    df_keywords = df_tmdb_keywords.copy()
    df_credits = df_tmdb_credits.copy()

    # Drop duplicates
    df_movie.drop_duplicates(subset=['id'], keep='first', inplace=True, ignore_index=True)
    df_keywords.drop_duplicates(subset=['id'], keep='first', inplace=True, ignore_index=True)
    df_credits.drop_duplicates(subset=['id'], keep='first', inplace=True, ignore_index=True)

    # Convert id to numeric
    df_movie['id'] = pd.to_numeric(df_movie['id'], errors='coerce')
    df_keywords['id'] = pd.to_numeric(df_keywords['id'], errors='coerce')
    df_credits['id'] = pd.to_numeric(df_credits['id'], errors='coerce')

    # Drop nan values
    df_movie.dropna(subset=['id'], inplace=True)
    df_keywords.dropna(subset=['id'], inplace=True)
    df_credits.dropna(subset=['id'], inplace=True)

    # Merge
    df_tmp = pd.merge(df_movie, df_keywords, on='id', how='inner')
    df_tmdb = pd.merge(df_tmp, df_credits, on='id', how='inner')

    return df_tmdb


def get_director(x):

    if isinstance(x, list):
        for i in x:
            if isinstance(i, dict) and i.get('job') == 'Director':
                return i.get('name', np.nan)
    return np.nan


def find_similar_rows(df, indices_different_box, columns):

    similar_rows = set()

    for idx in indices_different_box:
        row = df.loc[idx, columns]
        max_diff = row.max() * 0.05

        if all(abs(row - row.mean()) <= max_diff):
            similar_rows.add(idx)

    return list(similar_rows)


def clean_duplicates(df):
    print("Cleaning the columns...")
    #print("Creating new columns 'box_office_clean', 'release_date_clean and 'runtime_clean' with NaNs or NaTs")

    df['box_office_clean'] = float('nan')
    df['release_date_clean'] = pd.NaT
    df['runtime_clean'] = float('nan')

    #print(f"Each of them has respectively {df['box_office_clean'].isna().sum()}, {df['release_date_clean'].isna().sum()} "
    #      f"and {df['runtime_clean'].isna().sum()} NaN/Nat values")

    # All NaNs
    indices_all_nans_box = df.index[df[['box_office_revenue', 'revenue']].isna().all(axis=1)]
    indices_all_nans_date = df.index[df[['release_date_cmu', 'release_date_tmdb']].isna().all(axis=1)]
    indices_all_nans_time = df.index[df[['runtime_cmu', 'runtime_tmdb']].isna().all(axis=1)]

    #print(50*"-")
    #print(f"Number of rows where the two columns are NaNs: {len(indices_all_nans_box)}, {len(indices_all_nans_date)}, "
    #      f"{len(indices_all_nans_time)} respectively")

    #print("Some examples:")
    #print(df.loc[indices_all_nans_box[0]][['box_office_revenue', 'revenue']].head(2).to_string(), "\n")
    #print(df.loc[indices_all_nans_date[0], ['release_date_cmu', 'release_date_tmdb']], "\n")
    #print(df.loc[indices_all_nans_time[0], ['runtime_cmu', 'runtime_tmdb']])
    #print(50 * "-")

    # Same values
    indices_same_values_box = df.index[df['box_office_revenue'] == df['revenue']]
    indices_same_values_date = df.index[df['release_date_cmu'] == df['release_date_tmdb']]
    indices_same_values_time = df.index[df['runtime_cmu'] == df['runtime_tmdb']]

    #print(50 * "-")
    #print(f"Number of rows where the two columns have the same values: {len(indices_same_values_box)}, "
    #      f"{len(indices_same_values_date)}, {len(indices_same_values_time)} respectively\n")

    #print("Some examples:")
    #print(df.loc[indices_same_values_box[0], ['box_office_revenue', 'revenue']], "\n")
    #print(df.loc[indices_same_values_date[0], ['release_date_cmu', 'release_date_tmdb']], "\n")
    #print(df.loc[indices_same_values_time[0], ['runtime_cmu', 'runtime_tmdb']], "\n")

    #print("Updating values in the new columns...")
    df.loc[indices_same_values_box, 'box_office_clean'] = df.loc[indices_same_values_box, 'box_office_revenue']
    df.loc[indices_same_values_date, 'release_date_clean'] = df.loc[indices_same_values_date, 'release_date_cmu']
    df.loc[indices_same_values_time, 'runtime_clean'] = df.loc[indices_same_values_time, 'runtime_cmu']
    #print("Done\n")

    #print(f"Each of the column has now respectively {df['box_office_clean'].isna().sum()}, "
    #      f"{df['release_date_clean'].isna().sum()} and {df['runtime_clean'].isna().sum()} NaN/Nat values")
    #print(50 * "-")
    # One value
    indices_one_value_box = df.index[df[['box_office_revenue', 'revenue']].notna().sum(axis=1) == 1]
    indices_one_value_date = df.index[df[['release_date_cmu', 'release_date_tmdb']].notna().sum(axis=1) == 1]
    indices_one_value_time = df.index[df[['runtime_cmu', 'runtime_tmdb']].notna().sum(axis=1) == 1]

    #print(50 * "-")
    #print(f"Number of rows where the two columns have only one value: {len(indices_one_value_box)}, "
    #      f"{len(indices_one_value_date)}, {len(indices_one_value_time)} respectively\n")

    #print("Some examples:")
    #print(df.loc[indices_one_value_box[0], ['box_office_revenue', 'revenue']], "\n")
    #print(df.loc[indices_one_value_date[0], ['release_date_cmu', 'release_date_tmdb']], "\n")
    #print(df.loc[indices_one_value_time[0], ['runtime_cmu', 'runtime_tmdb']], "\n")

    #print("Updating values in the new columns...")
    for idx in indices_one_value_box:
        value = df.loc[idx, ['box_office_revenue', 'revenue']].dropna().values[0]
        df.at[idx, 'box_office_clean'] = value

    for idx in indices_one_value_box:
        value = df.loc[idx, ['box_office_revenue', 'revenue']].dropna().values[0]
        df.at[idx, 'box_office_clean'] = value

    for idx in indices_one_value_box:
        value = df.loc[idx, ['box_office_revenue', 'revenue']].dropna().values[0]
        df.at[idx, 'box_office_clean'] = value
    #print("Done")

    #print(f"Each of the column has now respectively {df['box_office_clean'].isna().sum()}, "
    #      f"{df['release_date_clean'].isna().sum()} and {df['runtime_clean'].isna().sum()} NaN/Nat values")
    #print(50 * "-")

    # Different values
    remaining_rows_box = df.shape[0] - len(indices_all_nans_box) - len(indices_same_values_box) - len(indices_one_value_box)
    remaining_rows_date = df.shape[0] - len(indices_all_nans_date) - len(indices_same_values_date) - len(indices_one_value_date)
    remaining_rows_time = df.shape[0] - len(indices_all_nans_time) - len(indices_same_values_time) - len(indices_one_value_time)
    #print(50 * "-")
    #print(f"Remaining number of rows for each column respectively: \n"
    #      f"{remaining_rows_box}, {remaining_rows_date}, {remaining_rows_time}")

    all_indices_box = np.concatenate([indices_all_nans_box, indices_same_values_box, indices_one_value_box])
    all_indices_date = np.concatenate([indices_all_nans_date, indices_same_values_date, indices_one_value_date])
    all_indices_time = np.concatenate([indices_all_nans_time, indices_same_values_time, indices_one_value_time])

    indices_different_box = df.index.difference(all_indices_box)
    indices_different_date = df.index.difference(all_indices_date)
    indices_different_time = df.index.difference(all_indices_time)

    # And what form do they have ?
    #print("Some examples:")
    #print(df.loc[indices_different_box, ['box_office_revenue', 'revenue']].head(), "\n")
    #print(df.loc[indices_different_date, ['release_date_cmu', 'release_date_tmdb']].head(), "\n")
    #print(df.loc[indices_different_time, ['runtime_cmu', 'runtime_tmdb']].head(), "\n")

    # We now clean the release date when we have different values
    #print("Now handling different values for the release date")
    indices_same_year = df.index[(df['release_date_cmu'].dt.year == df['release_date_tmdb'].dt.year)]
    indices_same_year = indices_same_year.difference(indices_same_values_date)

    #print(f"We have {len(indices_same_year)} rows with the same year but not equal. Some examples:\n")
    #print(df.loc[indices_same_year, ['release_date_cmu', 'release_date_tmdb']].head(), "\n")
    #print(f"Some examples of the remaining rows:\n")
    #print(df.loc[indices_different_date.difference(indices_same_year), ['release_date_cmu', 'release_date_tmdb']].head(), "\n")

    #print("Updating values in the release_date_clean...")
    for idx in indices_different_date:
        value = df.loc[idx, ['release_date_cmu', 'release_date_tmdb']].dropna().values[0]
        df.at[idx, 'release_date_clean'] = value
    #print("Done")
    #print(50 * "-")

    # Create boolean masks for the conditions
    indices_close_5_box = find_similar_rows(df, indices_different_box, columns=['box_office_revenue', 'revenue'])
    indices_close_5_time = find_similar_rows(df, indices_different_box, columns=['runtime_cmu', 'runtime_tmdb'])

    #print("Some examples with 5% difference:\n")
    #print(df.loc[indices_close_5_box, ['box_office_revenue', 'revenue']].head(), "\n")
    #print(df.loc[indices_close_5_time, ['runtime_cmu', 'runtime_tmdb']].head(), "\n")

    #print("Some examples with more difference:\n")
    #print(df.loc[indices_different_box.difference(indices_close_5_box), ['box_office_revenue', 'revenue']].head(), "\n")
    #print(df.loc[indices_different_time.difference(indices_close_5_time), ['runtime_cmu', 'runtime_tmdb']].head(), "\n")

    #print("Updating values in the box_office_clean and runtime_clean...")
    for idx in indices_different_box:
        value = df.loc[idx, ['box_office_revenue', 'revenue']].dropna().values[0]
        df.at[idx, 'box_office_clean'] = value

    for idx in indices_different_time:
        value = df.loc[idx, ['runtime_cmu', 'runtime_tmdb']].dropna().values[0]
        df.at[idx, 'box_office_clean'] = value
    #print("Done")
    print("All columns cleaned")

    return df


def create_world_map(data, col, world, log=False):

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    world.boundary.plot(ax=ax, linewidth=1, color="gray")

    im = data.plot(
        column=col,
        cmap='YlOrRd',
        legend=False,
        ax=ax,
        missing_kwds={"color": "lightgrey", "label": "No data"}
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    sm = plt.cm.ScalarMappable(cmap='YlOrRd',
                               norm=plt.Normalize(vmin=data[col].min(), vmax=data[col].max()))
    sm._A = []
    cbar = plt.colorbar(sm, cax=cax)
    if log:
        cbar.set_label('Logarithm Number of Movies', fontsize=14)
    else:
        cbar.set_label('Number of Movies', fontsize=14)

    plt.show()


def evaluate_model_for_user(userId, data):
    """
    Generates training and test splits for a given user ID, trains a neural network,
    and evaluates the model to return MSE and R^2 for both training and test sets.
    """
    # Filter and preprocess data for the given user
    user_rating = data[data['userId'] == userId]
    user_rating = user_rating.iloc[:, 5:].sample(frac=1)  # Shuffle data
    user_rating = user_rating.drop(columns=['timestamp'])
    y = user_rating['rating']
    x = user_rating.drop(columns='rating')

    # Split into training and testing sets
    length = len(y)
    x_size = int(length * 0.8)
    x_train, x_test = x.iloc[:x_size, :], x.iloc[x_size:, :]
    y_train, y_test = y.iloc[:x_size], y.iloc[x_size:]

    # Define the neural network model
    model = Sequential([
        Dense(64, input_dim=x_train.shape[1], activation='relu'),  # First hidden layer
        Dense(32, activation='relu'),  # Second hidden layer
        Dense(1, activation='linear')  # Output layer for regression
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    # Train the model
    model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

    # Predict on training and test sets
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Calculate evaluation metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Print evaluation results
    print(f"Evaluation for User ID {userId}:")
    print("Training Set Evaluation:")
    print(f"MSE: {train_mse:.4f}, R^2: {train_r2:.4f}")
    print("\nTest Set Evaluation:")
    print(f"MSE: {test_mse:.4f}, R^2: {test_r2:.4f}")

    # Return MSE values for both sets
    return train_mse, test_mse


def one_hot_encoding_genre(df,movies) :
    """
    Creates one hot encoded features per unique genre 

    """
    # Merges df and movies
    
    rating_films=pd.merge(movies, df, on='movieId', how='inner')

    # Create list of unique_genres
    unique_genres = movies['genres'].unique()
    all_genres = set('|'.join(unique_genres).split('|'))

    all_genres_list = list(all_genres)
    all_genres_list.remove('(no genres listed)')
    # Create a one hot encoding for each genre
    for genre in all_genres_list:
        rating_films[genre] = rating_films['genres'].apply(lambda x: genre in x.split('|'))
    return rating_films



def Total_eval(userId,df,movies) :
    """
    Creates one hot encoded features per unique genre 

    """
    # Merges df and movies
    mini_df=df[df['userId']==userId]
    rating_films=pd.merge(movies, mini_df, on='movieId', how='inner')
    
    # Create list of unique_genres
    unique_genres = movies['genres'].unique()
    all_genres = set('|'.join(unique_genres).split('|'))

    all_genres_list = list(all_genres)
    all_genres_list.remove('(no genres listed)')
    
    unique_director = movies['director'].unique()

    all_director_list = list(unique_director)
    all_director_list.remove('nobody')
    # Create a one hot encoding for each genre
    for genre in all_genres_list:
        rating_films[genre] = rating_films['genres'].apply(lambda x: genre in x.split('|'))
    # Create a one hot encoding for each director
    for director in all_director_list:
        rating_films[director] = rating_films['director'].apply(lambda x: director in x)
    return evaluate_model_for_user(userId,rating_films)