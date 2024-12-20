import re
import numpy as np
import pandas as pd
import calendar
import plotly.graph_objects as go
import plotly.express as px
from SPARQLWrapper import SPARQLWrapper, JSON
from collections import Counter


def format_imdb_id(imdb_id):
    # example of use: df_links['imdbId'] = df_links['imdbId'].apply(format_imdb_id)
    # '12836' --> 'tt0012836'
    return 'tt' + str(imdb_id).zfill(7)


def unformat_imdb_id(formatted_imdb_id):
    # example of use: df_links['imdbId'] = df_links['imdbId'].apply(unformat_imdb_id)
    # 'tt0012836' --> '12836'
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
    """
    Function to merge the dataframes of all datasets from TMdB together
    """
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

    df['box_office_clean'] = float('nan')
    df['release_date_clean'] = pd.NaT
    df['runtime_clean'] = float('nan')

    # All NaNs
    indices_all_nans_box = df.index[df[['box_office_revenue', 'revenue']].isna().all(axis=1)]
    indices_all_nans_date = df.index[df[['release_date_cmu', 'release_date_tmdb']].isna().all(axis=1)]
    indices_all_nans_time = df.index[df[['runtime_cmu', 'runtime_tmdb']].isna().all(axis=1)]

    # Same values
    indices_same_values_box = df.index[df['box_office_revenue'] == df['revenue']]
    indices_same_values_date = df.index[df['release_date_cmu'] == df['release_date_tmdb']]
    indices_same_values_time = df.index[df['runtime_cmu'] == df['runtime_tmdb']]

    df.loc[indices_same_values_box, 'box_office_clean'] = df.loc[indices_same_values_box, 'box_office_revenue']
    df.loc[indices_same_values_date, 'release_date_clean'] = df.loc[indices_same_values_date, 'release_date_cmu']
    df.loc[indices_same_values_time, 'runtime_clean'] = df.loc[indices_same_values_time, 'runtime_cmu']

    # One value
    indices_one_value_box = df.index[df[['box_office_revenue', 'revenue']].notna().sum(axis=1) == 1]
    indices_one_value_date = df.index[df[['release_date_cmu', 'release_date_tmdb']].notna().sum(axis=1) == 1]
    indices_one_value_time = df.index[df[['runtime_cmu', 'runtime_tmdb']].notna().sum(axis=1) == 1]

    for idx in indices_one_value_box:
        value = df.loc[idx, ['box_office_revenue', 'revenue']].dropna().values[0]
        df.at[idx, 'box_office_clean'] = value

    for idx in indices_one_value_box:
        value = df.loc[idx, ['box_office_revenue', 'revenue']].dropna().values[0]
        df.at[idx, 'box_office_clean'] = value

    for idx in indices_one_value_box:
        value = df.loc[idx, ['box_office_revenue', 'revenue']].dropna().values[0]
        df.at[idx, 'box_office_clean'] = value

    # Different values
    remaining_rows_box = df.shape[0] - len(indices_all_nans_box) - len(indices_same_values_box) - len(indices_one_value_box)
    remaining_rows_date = df.shape[0] - len(indices_all_nans_date) - len(indices_same_values_date) - len(indices_one_value_date)
    remaining_rows_time = df.shape[0] - len(indices_all_nans_time) - len(indices_same_values_time) - len(indices_one_value_time)

    all_indices_box = np.concatenate([indices_all_nans_box, indices_same_values_box, indices_one_value_box])
    all_indices_date = np.concatenate([indices_all_nans_date, indices_same_values_date, indices_one_value_date])
    all_indices_time = np.concatenate([indices_all_nans_time, indices_same_values_time, indices_one_value_time])

    indices_different_box = df.index.difference(all_indices_box)
    indices_different_date = df.index.difference(all_indices_date)
    indices_different_time = df.index.difference(all_indices_time)

    # We now clean the release date when we have different values
    indices_same_year = df.index[(df['release_date_cmu'].dt.year == df['release_date_tmdb'].dt.year)]
    indices_same_year = indices_same_year.difference(indices_same_values_date)

    for idx in indices_different_date:
        value = df.loc[idx, ['release_date_cmu', 'release_date_tmdb']].dropna().values[0]
        df.at[idx, 'release_date_clean'] = value

    # Create boolean masks for the conditions
    indices_close_5_box = find_similar_rows(df, indices_different_box, columns=['box_office_revenue', 'revenue'])
    indices_close_5_time = find_similar_rows(df, indices_different_box, columns=['runtime_cmu', 'runtime_tmdb'])

    for idx in indices_different_box:
        value = df.loc[idx, ['box_office_revenue', 'revenue']].dropna().values[0]
        df.at[idx, 'box_office_clean'] = value

    for idx in indices_different_time:
        value = df.loc[idx, ['runtime_cmu', 'runtime_tmdb']].dropna().values[0]
        df.at[idx, 'box_office_clean'] = value
    print("All columns cleaned !")

    return df


def plot_movies_by_country(df):
    """
    Create a map showing the number of movies per country
    with logarithmic-like color scaling, but true values on the colorbar
    """
    if 'country' not in df.columns or 'movie_count' not in df.columns:
        raise ValueError("DataFrame must have 'country' and 'movie_count' columns")

    # Apply logarithmic transformation for color scaling
    df['log_movie_count'] = np.log1p(df['movie_count'])

    # Define tick values and labels for the colorbar
    tick_vals = np.linspace(df['log_movie_count'].min(), df['log_movie_count'].max(), num=6)
    tick_labels = [int(np.expm1(val)) for val in tick_vals]

    fig = go.Figure(data=go.Choropleth(
        locations=df['country'],
        locationmode='country names',
        z=df['log_movie_count'],
        text=[f"{country}: {count:,} movies" for country, count in zip(df['country'], df['movie_count'])],
        hoverinfo="text",
        colorscale='Viridis',
        colorbar=dict(
            title='Number of Movies',
            tickvals=tick_vals,
            ticktext=[f"{label:,}" for label in tick_labels]
        )
    ))

    fig.update_layout(
        geo_scope='world',
        height=600,
        width=1000,
        title_x=0.5
    )

    return fig


def plot_language_count(df):
    fig = px.bar(
        df,
        x='language',
        y='count',
        title='Number of Films per Language',
        labels={'language': 'Language', 'count': 'Count'},
        log_y=True,
        color='language',
    )

    fig.update_layout(
        width=1100,
        height=600,
        xaxis=dict(title='Language', tickangle=30, title_font=dict(size=14)),
        yaxis=dict(
            title='Count (Log Scale)',
            title_font=dict(size=14),
            tickfont=dict(size=14),
        ),
        title_font=dict(size=18),
        legend_title=None
    )
    return fig


def plot_directors_count(df):
    fig = px.bar(
        df,
        x='director',
        y='Occurrences',
        title='Number of Films per Director',
        labels={'director': 'Director', 'Occurrences': 'Occurrences'},
        log_y=True,
        color='director',
    )

    fig.update_layout(
        width=1100,
        height=600,
        xaxis=dict(
            title='Director',
            tickangle=30,
            title_font=dict(size=14),
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title='Occurrences',
            title_font=dict(size=14),
            tickfont=dict(size=14),
        ),
        title_font=dict(size=18),
        legend_title=None,
    )
    return fig


def plot_actors_count(df):
    fig = px.bar(
        df,
        x='Actor',
        y='Occurrences',
        title='Number of Films per Actor',
        labels={'Actor': 'Actor', 'Occurrences': 'Occurrences'},
        log_y=True,
        color='Actor',
    )

    fig.update_layout(
        width=1100,
        height=600,
        xaxis=dict(
            title='Actor',
            tickangle=30,
            title_font=dict(size=14),
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title='Occurrences',
            title_font=dict(size=14),
            tickfont=dict(size=14),
        ),
        title_font=dict(size=18),
        legend_title=None,
    )
    return fig


def plot_year_occurences(df):
    fig = go.Figure(data=[go.Bar(x=df.index, y=df.values)])

    fig.update_layout(
        title='Number of movies by Year',
        xaxis_title='Year',
        yaxis_title='Number of movies in total',
        xaxis=dict(tickangle=0),
        height=600,
        width=1000
    )
    return fig


def plot_month_occurences(df):
    month_names = [calendar.month_name[month] for month in
                   df.index.astype(int)]  # Convert month indices to names

    fig = go.Figure(data=[go.Bar(
        x=month_names,
        y=df.values,
        text=df.values,
        textposition='outside'
    )])

    fig.update_layout(
        title='Number of Movies by Month',
        xaxis_title='Month',
        yaxis_title='Number of Movies in Total',
        xaxis=dict(tickangle=-30),
        height=600,
        width=1000
    )
    return fig


def plot_day_occurences(df):
    fig = go.Figure(data=[go.Bar(x=df.index, y=df.values,
                                 text=df.values, textposition='outside')])

    fig.update_layout(
        title='Number of movies by Day',
        xaxis_title='Day',
        yaxis_title='Number of movies in total',
        xaxis=dict(tickangle=0),
        height=600,
        width=1000
    )
    return fig


def value_counts_for_genre(genre_column):
    genre_counts = {}
    for row in genre_column:
        if isinstance(row, str):
            for genre in row.split(","):
                genre = genre.strip()
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

    # Create a DataFrame from genre counts
    genre_counts_df = pd.DataFrame(
        list(genre_counts.items()), columns=["Genre", "Occurrences"]
    ).sort_values(by="Occurrences", ascending=False)

    top_genres = genre_counts_df.iloc[:20]

    return top_genres


def plot_genre_counts(df):
    fig = px.bar(
        df,
        x='Genre',
        y='Occurrences',
        title='Number of Films per Genre (Logarithmic Scale)',
        labels={'Genre': 'Genre', 'Occurrences': 'Occurrences'},
        log_y=True,
        color='Genre',
    )

    fig.update_layout(
        width=1100,
        height=600,
        xaxis=dict(
            title='Genre',
            tickangle=30,
            title_font=dict(size=14),
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title='Occurrences (Log Scale)',
            title_font=dict(size=14),
            tickfont=dict(size=14),
        ),
        title_font=dict(size=18),
        legend_title=None,
    )
    return fig


def value_counts_for_keywords(keywords_column):
    all_keywords = [word for keywords_list in keywords_column for word in keywords_list]
    all_keywords_alt = keywords_column.explode().tolist()
    keyword_counts = Counter(all_keywords)
    most_common_keywords = keyword_counts.most_common()
    keyword_freq_series = pd.Series(keyword_counts).sort_values(ascending=False)
    return keyword_freq_series


def plot_runtime_over_time(df):
    fig = px.line(df, x='year', y='runtime_clean',
                  title='Average Runtime of Movies Over Time',
                  labels={'year': 'Year', 'runtime_clean': 'Average Runtime (minutes)'})

    fig.update_layout(
        title_font_size=20,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        template='plotly_white'
    )
    return fig


# ################################ VISU RATINGS ################################

def plot_hist_ratings(df):
    mean_rating = df['rating'].mean()

    fig = px.histogram(
        df,
        x='rating',
        nbins=len(df['rating'].unique()),
        title='Histogram of Ratings',
        labels={'rating': 'Ratings'},
        color_discrete_sequence=['skyblue']
    )

    fig.add_trace(
        go.Scatter(
            x=[mean_rating, mean_rating],
            y=[0, df['rating'].value_counts().max()],
            mode='lines',
            name=f'Mean Rating: {mean_rating:.2f}',
            line=dict(color='red', width=2, dash='dash')
        )
    )

    fig.update_layout(
        xaxis_title='Ratings',
        yaxis_title='Frequency',
        bargap=0.02,
        height=600,
        width=1000,
        template='plotly_white'
    )
    return fig


def plot_most_rated_movies(df, df_names):
    rating_counts = df['imdbId'].value_counts().head(10).reset_index()
    rating_counts.columns = ['imdbId', 'rating_count']

    df_names['formatted_imdb_id'] = df_names['imdb_id'].apply(unformat_imdb_id)

    names = df_names[df_names['formatted_imdb_id'].isin(rating_counts['imdbId'])][['title', 'formatted_imdb_id']]
    merged = rating_counts.merge(
        names,
        left_on='imdbId',
        right_on='formatted_imdb_id',
        how='left'
    )

    merged['imdbId'] = merged['title']

    fig = px.bar(
        merged,
        x='imdbId',
        y='rating_count',
        title='Top 10 Movies by Number of Ratings',
        labels={'imdbId': 'Movie Name', 'rating_count': 'Number of Ratings'},
        color_discrete_sequence=['orange']
    )

    fig.update_layout(
        title_font_size=20,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        template='plotly_white'
    )
    return fig


def plot_most_ratings_users(df):
    user_stats = df['userId'].value_counts().head(10).reset_index()
    user_stats.columns = ['userId', 'rating_count']

    user_stats['userId'] = user_stats['userId'].astype(str)

    fig = px.bar(
        user_stats,
        x='userId',
        y='rating_count',
        title='Top 10 Users by Number of Ratings',
        labels={'userId': 'User ID', 'rating_count': 'Number of Ratings'},
        color_discrete_sequence=['red']
    )

    fig.update_layout(
        title_font_size=20,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        xaxis=dict(type='category'),
        template='plotly_white'
    )
    return fig


def plot_average_vs_number_of_ratings(df):
    movie_stats = df.groupby('imdbId').agg(
        average_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()

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

    fig.update_layout(
        title_font_size=20,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        template='plotly_white',
        height=600,
        width=1000
    )

    fig.update_xaxes(type='log')
    return fig


def remove_actors_one_appearance(cast_column):

    actors_exploded = cast_column.explode()
    actors_exploded = actors_exploded.dropna()
    actor_counts = actors_exploded.value_counts()
    actors_to_keep = actor_counts[actor_counts > 1].index

    actor_counts_df = actor_counts.reset_index()
    actor_counts_df.columns = ['Actor', 'Occurrences']

    filtered_cast_column = cast_column.apply(
        lambda cast_list: [actor for actor in cast_list if actor in actors_to_keep]
    )
    return filtered_cast_column, actor_counts_df


def remove_directors_one_movie(df_final_dataset):
    director_column = df_final_dataset['director']

    director_counts = director_column.value_counts()
    director_to_keep = director_counts[director_counts > 1].index

    director_counts_df = director_counts.reset_index()
    director_counts_df.columns = ['director', 'Occurrences']

    filtered_director_director = df_final_dataset[director_column.isin(director_to_keep)][
        'director']

    return filtered_director_director, director_counts_df


