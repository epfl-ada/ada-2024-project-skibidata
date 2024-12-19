import time
import re
import ast
import numpy as np
import pandas as pd
import calendar
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import  plotly.express as px
from SPARQLWrapper import SPARQLWrapper, JSON
from collections import Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
"""
from sklearn.metrics import mean_squared_error, r2_score

def format_imdb_id(imdb_id):
    # example of use: df_links['imdbId'] = df_links['imdbId'].apply(format_imdb_id)
    return 'tt' + str(imdb_id).zfill(7)


def unformat_imdb_id(formatted_imdb_id):
    # example of use: df_links['imdbId'] = df_links['imdbId'].apply(unformat_imdb_id)
    return formatted_imdb_id[2:].lstrip('0')


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

"""
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
"""
'''
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
'''


def plot_movies_by_country(df):
    """
    Create a choropleth map showing the number of movies per country
    with logarithmic-like color scaling, but true values on the colorbar
    and in the hover tooltip.

    Parameters:
    df (pandas.DataFrame): DataFrame with columns 'country' and 'movie_count'

    Returns:
    plotly.graph_objs._figure.Figure: Interactive choropleth map
    """
    # Ensure the DataFrame has the correct columns
    if 'country' not in df.columns or 'movie_count' not in df.columns:
        raise ValueError("DataFrame must have 'country' and 'movie_count' columns")

    # Apply logarithmic transformation for color scaling
    df['log_movie_count'] = np.log1p(df['movie_count'])  # Avoid log(0) issues

    # Define tick values and labels for the colorbar
    tick_vals = np.linspace(df['log_movie_count'].min(), df['log_movie_count'].max(), num=6)
    tick_labels = [int(np.expm1(val)) for val in tick_vals]  # Reverse the log1p transformation

    # Create the choropleth map
    fig = go.Figure(data=go.Choropleth(
        locations=df['country'],  # Country names
        locationmode='country names',  # Use full country names
        z=df['log_movie_count'],  # Logarithmically transformed values for the color scale
        text=[f"{country}: {count:,} movies" for country, count in zip(df['country'], df['movie_count'])],  # Hover text
        hoverinfo="text",  # Use custom hover text
        colorscale='Viridis',  # Color gradient
        colorbar=dict(
            title='Number of Movies',
            tickvals=tick_vals,  # Colorbar ticks based on the transformed scale
            ticktext=[f"{label:,}" for label in tick_labels]  # Display original values as tick labels
        )
    ))

    # Customize the layout
    fig.update_layout(
        # title_text='Number of Movies per Country',
        geo_scope='world',  # Set the map scope to world
        height=600,  # Map height
        width=1000,  # Map width
        title_x=0.5  # Center the title
    )

    return fig


def plot_language_count(df):
    # Create the plot
    fig = px.bar(
        df,
        x='language',
        y='count',
        title='Number of Films per Language',
        labels={'language': 'Language', 'count': 'Count'},
        log_y=True,  # Logarithmic scale for the y-axis
        color='language',  # Color bars by language
    )

    # Customize the layout and make the plot larger
    fig.update_layout(
        width=1100,  # Set width to make the plot larger
        height=600,  # Set height to make the plot larger
        xaxis=dict(title='Language', tickangle=30, title_font=dict(size=14)),
        yaxis=dict(
            title='Count (Log Scale)',
            title_font=dict(size=14),
            tickfont=dict(size=14),  # Increase y-axis tick font size
        ),
        title_font=dict(size=18),
        legend_title=None
    )
    return fig


def plot_directors_count(df):
    # Create the plot
    fig = px.bar(
        df,
        x='director',
        y='Occurrences',
        title='Number of Films per Director',
        labels={'director': 'Director', 'Occurrences': 'Occurrences'},
        log_y=True,  # Logarithmic scale for the y-axis
        color='director',  # Color bars by director
    )

    # Customize the layout
    fig.update_layout(
        width=1100,  # Set width to make the plot larger
        height=600,  # Set height to make the plot larger
        xaxis=dict(
            title='Director',
            tickangle=30,  # Rotate x-axis labels
            title_font=dict(size=14),
            tickfont=dict(size=10),  # Smaller tick font size for directors
        ),
        yaxis=dict(
            title='Occurrences',
            title_font=dict(size=14),
            tickfont=dict(size=14),  # Larger y-axis tick font size
        ),
        title_font=dict(size=18),
        legend_title=None,  # Remove legend title
    )
    return fig


def plot_actors_count(df):

    fig = px.bar(
        df,
        x='Actor',
        y='Occurrences',
        title='Number of Films per Actor',
        labels={'Actor': 'Actor', 'Occurrences': 'Occurrences'},
        log_y=True,  # Logarithmic scale for the y-axis
        color='Actor',  # Color bars by actor
    )

    # Customize the layout
    fig.update_layout(
        width=1100,  # Set width to make the plot larger
        height=600,  # Set height to make the plot larger
        xaxis=dict(
            title='Actor',
            tickangle=30,  # Rotate x-axis labels
            title_font=dict(size=14),
            tickfont=dict(size=10),  # Smaller tick font size for actors
        ),
        yaxis=dict(
            title='Occurrences',
            title_font=dict(size=14),
            tickfont=dict(size=14),  # Larger y-axis tick font size
        ),
        title_font=dict(size=18),
        legend_title=None,  # Remove legend title
    )

    return fig


def plot_year_occurences(df):
    fig = go.Figure(data=[go.Bar(x=df.index, y=df.values)])

    # Update layout for labels and title
    fig.update_layout(
        title='Number of movies by Year',
        xaxis_title='Year',
        yaxis_title='Number of movies in total',
        xaxis=dict(tickangle=0),
        height=600,
        width=1000  # Rotate x-axis labels if needed
    )

    # Show the plot
    return fig

def plot_month_occurences(df):
    month_names = [calendar.month_name[month] for month in
                   df.index.astype(int)]  # Convert month indices to names

    # Create the bar chart
    fig = go.Figure(data=[go.Bar(
        x=month_names,  # Use month names instead of indices
        y=df.values,
        text=df.values,  # Add text (values) on top of the bars
        textposition='outside'  # Position the text outside the bars
    )])

    # Update layout for labels and title
    fig.update_layout(
        title='Number of Movies by Month',
        xaxis_title='Month',
        yaxis_title='Number of Movies in Total',
        xaxis=dict(tickangle=-30),  # Rotate x-axis labels if necessary
        height=600,
        width=1000
    )

    return fig


def plot_day_occurences(df):
    fig = go.Figure(data=[go.Bar(x=df.index, y=df.values,
                                 text=df.values, textposition='outside')])

    # Update layout for labels and title
    fig.update_layout(
        title='Number of movies by Day',
        xaxis_title='Day',
        yaxis_title='Number of movies in total',
        xaxis=dict(tickangle=0),
        height=600,
        width=1000  # Rotate x-axis labels if needed
    )

    return fig


def value_counts_for_genre(genre_column):
    genre_counts = {}
    for row in genre_column:
        if isinstance(row, str):  # Ensure the row is a string
            for genre in row.split(","):
                genre = genre.strip()
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

    # Create a DataFrame from genre counts
    genre_counts_df = pd.DataFrame(
        list(genre_counts.items()), columns=["Genre", "Occurrences"]
    ).sort_values(by="Occurrences", ascending=False)

    # Step 2: Select Top Genres (e.g., Top 20)
    top_genres = genre_counts_df.iloc[:20]

    return top_genres


def plot_genre_counts(df):
    # Step 3: Create the Plot
    fig = px.bar(
        df,
        x='Genre',
        y='Occurrences',
        title='Number of Films per Genre (Logarithmic Scale)',
        labels={'Genre': 'Genre', 'Occurrences': 'Occurrences'},
        log_y=True,  # Apply logarithmic scale
        color='Genre',  # Color bars by genre
    )

    # Step 4: Customize the Layout
    fig.update_layout(
        width=1100,  # Set plot width
        height=600,  # Set plot height
        xaxis=dict(
            title='Genre',
            tickangle=30,  # Rotate x-axis labels for better visibility
            title_font=dict(size=14),
            tickfont=dict(size=10),  # Smaller font for genres
        ),
        yaxis=dict(
            title='Occurrences (Log Scale)',
            title_font=dict(size=14),
            tickfont=dict(size=14),  # Larger y-axis tick font size
        ),
        title_font=dict(size=18),
        legend_title=None,  # Remove legend title
    )

    return fig


def value_counts_for_keywords(keywords_column):
    # Method 1: Using list comprehension
    all_keywords = [word for keywords_list in keywords_column for word in keywords_list]

    # Method 2: Using pandas explode
    all_keywords_alt = keywords_column.explode().tolist()
    # Count the frequencies of keywords
    keyword_counts = Counter(all_keywords)

    # If you want to see the most common keywords
    most_common_keywords = keyword_counts.most_common()  # Returns list of (word, count) tuples
    # Or if you prefer a pandas Series
    keyword_freq_series = pd.Series(keyword_counts).sort_values(ascending=False)

    return keyword_freq_series


def plot_runtime_over_time(df):
    # Create the line plot using Plotly
    fig = px.line(df, x='year', y='runtime_clean',
                  title='Average Runtime of Movies Over Time',
                  labels={'year': 'Year', 'runtime_clean': 'Average Runtime (minutes)'})

    # Update the layout for better visualization
    fig.update_layout(
        title_font_size=20,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        template='plotly_white'
    )

    return fig


# ################################ VISU RATINGS ################################

def plot_hist_ratings(df):
    # Compute the histogram using Plotly
    mean_rating = df['rating'].mean()

    fig = px.histogram(
        df,
        x='rating',
        nbins=len(df['rating'].unique()),  # One bin per unique rating
        title='Histogram of Ratings',
        labels={'rating': 'Ratings'},
        color_discrete_sequence=['skyblue']
    )

    # Add a vertical line for the mean rating
    fig.add_trace(
        go.Scatter(
            x=[mean_rating, mean_rating],
            y=[0, df['rating'].value_counts().max()],
            mode='lines',
            name=f'Mean Rating: {mean_rating:.2f}',
            line=dict(color='red', width=2, dash='dash')
        )
    )

    # Customize the layout
    fig.update_layout(
        xaxis_title='Ratings',
        yaxis_title='Frequency',
        bargap=0.02,  # Reduced gap for tighter bars
        height=600,
        width=1000,  # Increased height
        template='plotly_white'
    )

    return fig


def plot_most_rated_movies(df, df_names):
    # Get the top 10 movies by the number of ratings
    rating_counts = df['imdbId'].value_counts().head(10).reset_index()
    rating_counts.columns = ['imdbId', 'rating_count']

    # Ensure 'imdb_id' is formatted correctly for matching
    df_names['formatted_imdb_id'] = df_names['imdb_id'].apply(unformat_imdb_id)

    # Merge names and rating counts
    names = df_names[df_names['formatted_imdb_id'].isin(rating_counts['imdbId'])][['title', 'formatted_imdb_id']]
    merged = rating_counts.merge(
        names,
        left_on='imdbId',
        right_on='formatted_imdb_id',
        how='left'
    )

    # Replace movie IDs with titles for the x-axis
    merged['imdbId'] = merged['title']

    # Create the bar chart using Plotly
    fig = px.bar(
        merged,
        x='imdbId',
        y='rating_count',
        title='Top 10 Movies by Number of Ratings',
        labels={'imdbId': 'Movie Name', 'rating_count': 'Number of Ratings'},
        color_discrete_sequence=['orange']
    )

    # Update the layout for better visualization
    fig.update_layout(
        title_font_size=20,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        template='plotly_white'
    )

    return fig


def plot_most_ratings_users(df):
    # Get the top 10 users by number of ratings
    user_stats = df['userId'].value_counts().head(10).reset_index()
    user_stats.columns = ['userId', 'rating_count']

    # Convert userId to a string to treat it as a categorical variable
    user_stats['userId'] = user_stats['userId'].astype(str)

    # Create the bar chart using Plotly
    fig = px.bar(
        user_stats,
        x='userId',
        y='rating_count',
        title='Top 10 Users by Number of Ratings',
        labels={'userId': 'User ID', 'rating_count': 'Number of Ratings'},
        color_discrete_sequence=['red']
    )

    # Update the layout for better visualization
    fig.update_layout(
        title_font_size=20,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        xaxis=dict(type='category'),  # Ensure x-axis is categorical
        template='plotly_white'
    )

    return fig


def plot_average_vs_number_of_ratings(df):
    # Compute the movie statistics
    movie_stats = df.groupby('imdbId').agg(
        average_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()

    # Create the scatter plot using Plotly
    fig = px.scatter(
        movie_stats,
        x='num_ratings',
        y='average_rating',
        size_max=8,
        opacity=0.6,
        color_discrete_sequence=['purple'],
        title='Average Rating vs. Number of Ratings',
        labels={'num_ratings': 'Number of Ratings', 'average_rating': 'Average Rating'}
    )

    # Customize the axes and layout
    fig.update_layout(
        title_font_size=20,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        template='plotly_white',
        height=600,
        width=1000
    )

    # Set x-axis to logarithmic scale
    fig.update_xaxes(type='log')

    return fig


