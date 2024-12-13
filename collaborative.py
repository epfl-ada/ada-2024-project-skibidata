import time
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heapq
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


def format_imdb_id(imdb_id):
    # example of use: df_links['imdbId'] = df_links['imdbId'].apply(format_imdb_id)
    return 'tt' + str(imdb_id).zfill(7)


def unformat_imdb_id(formatted_imdb_id):
    # example of use: df_links['imdbId'] = df_links['imdbId'].apply(unformat_imdb_id)
    return formatted_imdb_id[2:].lstrip('0')


def filter_ratings_dataframe(df, k):
    """
    Function that remove users and movies with less than k ratings
    """
    # Remove Users
    review_counts = df.groupby('userId')['rating'].count()
    valid_users = review_counts[review_counts >= k].index
    print(f"Number of valid users: {len(valid_users)}")
    df_filtered = df[df['userId'].isin(valid_users)].copy()

    # Remove Users
    review_counts = df_filtered.groupby('movieId')['rating'].count()
    valid_movies = review_counts[review_counts >= 5].index
    print(f"Number of valid movies: {len(valid_movies)}")
    df_ratings = df_filtered[df_filtered['movieId'].isin(valid_movies)].copy()

    return df_ratings


def create_mapping(column_of_df):
    """Create mappings between original and matrix indices with logging."""
    start_time = time.time()
    print("\nCreating ID mappings...")

    unique_elements = column_of_df.unique()

    map_ = {element: idx for idx, element in enumerate(unique_elements)}
    reverse_map = {idx: element for element, idx in map_.items()}

    print(f"Mapping created in {time.time() - start_time:.2f} seconds")
    print(f"Total unique elements: {len(unique_elements)}")

    return map_, reverse_map


def create_sparse_ratings_matrix(df_ratings, n_users, n_items):
    # Transform DataFrame to sparse matrix
    rows = [user for user in df_ratings['sparse_user_id']]
    cols = [movie for movie in df_ratings['sparse_movie_id']]
    ratings = df_ratings['rating'].values

    ratings_matrix = csr_matrix(
        (ratings, (rows, cols)),
        shape=(n_users, n_items)
    )

    # Compute global mean
    global_mean = df_ratings['rating'].mean()
    print(f"Global mean rating: {global_mean:.2f}")

    return ratings_matrix


def split_into_train_and_test(df, test_size=0.2, random_state=42):
    """
    Split ratings DataFrame into train and test sets while preserving user-level splits.
    """
    print("\n--- Performing Train-Test Split ---")

    splits = [
        train_test_split(user_ratings, test_size=test_size, random_state=random_state)
        for _, user_ratings in df.groupby('sparse_user_id')
    ]

    train_data, test_data = zip(*splits)

    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)

    print(f"Total ratings: {len(df)}")
    print(f"Training set size: {len(train_df)} ({len(train_df) / len(df) * 100:.2f}%)")
    print(f"Test set size: {len(test_df)} ({len(test_df) / len(df) * 100:.2f}%)")

    return train_df, test_df


################################ Neighbordhood-based Methods ################################
def optimize_recommendation_calculation(test, df_ratings, weights, indices, num_users=100, k=30):
    np.random.seed(42)
    unique_users = test['sparse_user_id'].unique()
    selected_users = np.random.choice(unique_users, size=num_users, replace=False)

    # Create weights map once
    weights_map = dict(zip(indices, weights))

    # Precompute movie-user mapping
    movie_users = test.groupby('sparse_movie_id')['sparse_user_id'].agg(list)

    # Precompute user mean and std ratings for each user's own ratings
    user_mean_ratings = df_ratings.groupby('sparse_user_id')['rating'].mean()
    user_std_ratings = df_ratings.groupby('sparse_user_id')['rating'].std().fillna(1)

    results = {user: {} for user in selected_users}

    for user in selected_users:
        # Compute user's own mean and std of ratings
        user_ratings_subset = df_ratings[df_ratings['sparse_user_id'] == user]
        r_u = user_ratings_subset['rating'].mean()
        std_u = user_ratings_subset['rating'].std()

        user_movies = test[test['sparse_user_id'] == user]['sparse_movie_id'].values
        user_results_mean_centering = []
        user_results_z_normalization = []
        user_results_basic = []

        for movie in user_movies:
            # Get top K weighted users for this movie
            movie_users_list = movie_users.get(movie, [])
            top_users = heapq.nlargest(
                k,
                [(uid, weights_map.get(uid, 0)) for uid in movie_users_list if weights_map.get(uid, 0) > 0],
                key=lambda x: x[1]
            )

            # Filter ratings for top users
            user_ratings = df_ratings[
                (df_ratings['sparse_movie_id'] == movie) &
                (df_ratings['sparse_user_id'].isin([u[0] for u in top_users]))
                ].copy()

            if len(user_ratings) == 0:
                continue

            # Add columns using .loc to avoid warnings
            user_ratings.loc[:, 'user_weight'] = user_ratings['sparse_user_id'].map(dict(top_users))
            user_ratings.loc[:, 'mean_rating'] = user_ratings['sparse_user_id'].map(user_mean_ratings)
            user_ratings.loc[:, 'std_rating'] = user_ratings['sparse_user_id'].map(user_std_ratings)
            user_ratings.loc[:, 'weighted_diff'] = user_ratings['user_weight'] * (
                        user_ratings['rating'] - user_ratings['mean_rating'])

            numerator = user_ratings['weighted_diff'].sum()
            denominator = user_ratings['user_weight'].abs().sum()
            numerator2 = (user_ratings['weighted_diff'] / user_ratings['std_rating']).sum()
            numerator3 = (user_ratings['user_weight'] * user_ratings['rating']).sum()
            # print("Numerator\n:", numerator)
            # print("Numerator2\n:", numerator2)
            # print("Numerator3\n:", numerator3)

            if denominator > 0:
                result = numerator / denominator + r_u
                result2 = numerator2 * std_u / denominator + r_u
                result3 = numerator3 / denominator

                user_results_mean_centering.append((movie, result))
                user_results_z_normalization.append((movie, result2))
                user_results_basic.append((movie, result3))

        results[user]['basic'] = user_results_basic
        results[user]['mean_centering'] = user_results_mean_centering
        results[user]['z_normalization'] = user_results_z_normalization

    return results

def add_predicted_ratings(results, selected_users_df):
    # Create a copy of the DataFrame to avoid modifying the original
    df = selected_users_df.copy()

    # Create prediction lookup dictionaries for each method
    prediction_lookups = {
        'basic': {},
        'mean_centering': {},
        'z_normalization': {}
    }

    for user, method_results in results.items():
        for method, movie_ratings in method_results.items():
            prediction_lookups[method][user] = dict(movie_ratings)

    # Vectorized prediction function for each method
    def get_predicted_rating(user, movie, method):
        # Return a single float value or None
        return prediction_lookups[method].get(user, {}).get(movie)

    # Add predicted ratings for each method
    for method in ['basic', 'mean_centering', 'z_normalization']:
        # Add predicted rating column
        df[f'predicted_rating_{method}'] = df.apply(
            lambda row: get_predicted_rating(row['sparse_user_id'], row['sparse_movie_id'], method),
            axis=1
        )

        # Add error column for each method
        # Use .fillna() to handle cases where prediction is None
        df[f'rating_error_{method}'] = np.abs(df['rating'] - df[f'predicted_rating_{method}'].fillna(np.nan))
        df[f'rating_error_squared_{method}'] = (df['rating'] - df[f'predicted_rating_{method}'].fillna(np.nan)) ** 2

    # Drop rows with no predictions across all methods
    df_with_predictions = df.dropna(
        subset=[
            'predicted_rating_basic',
            'predicted_rating_mean_centering',
            'predicted_rating_z_normalization'
        ]
    )

    return df_with_predictions


def plot_predicted_ratings_distribution(df):
    plt.figure(figsize=(15, 5))
    for i, method in enumerate(['basic', 'mean_centering', 'z_normalization'], 1):
        plt.subplot(1, 3, i)
        plt.hist(df[f'rating_error_{method}'], bins=30, edgecolor='black')
        plt.title(f'{method.replace("_", " ").title()} Error Distribution')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()






############################# Dimensionality Reduction Methods ##############################
class SparseSVDRecommender:
    def __init__(self, n_factors: int = 100, learning_rate: float = 0.005,
                 regularization: float = 0.02, n_epochs: int = 20):
        """
        Initialize the SVD recommender system with verbose logging and serialization.
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs

        # Model parameters
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None

        # Mapping dictionaries
        self.user_id_map = {}
        self.movie_id_map = {}
        self.reverse_user_id_map = {}
        self.reverse_movie_id_map = {}

    def fit(self, ratings_matrix_train: csr_matrix,
            user_id_map: dict = None,
            movie_id_map: dict = None) -> 'SparseSVDRecommender3':
        """
        Train the model using a pre-computed sparse ratings matrix.

        Args:
            ratings_matrix_train (csr_matrix): Sparse training ratings matrix
            user_id_map (dict, optional): Mapping of original user IDs to matrix indices
            movie_id_map (dict, optional): Mapping of original movie IDs to matrix indices
        """
        print("\n--- Starting SVD Training ---")
        start_total_time = time.time()

        # If maps are not provided, create default mappings
        if user_id_map is None:
            self.user_id_map = {i: i for i in range(ratings_matrix_train.shape[0])}
            self.reverse_user_id_map = self.user_id_map
        else:
            self.user_id_map = user_id_map
            self.reverse_user_id_map = {idx: user for user, idx in user_id_map.items()}

        if movie_id_map is None:
            self.movie_id_map = {i: i for i in range(ratings_matrix_train.shape[1])}
            self.reverse_movie_id_map = self.movie_id_map
        else:
            self.movie_id_map = movie_id_map
            self.reverse_movie_id_map = {idx: movie for movie, idx in movie_id_map.items()}

            # Compute global mean
        self.global_mean = ratings_matrix_train.data.mean()
        print(f"Global mean rating: {self.global_mean:.2f}")

        # Get matrix dimensions
        n_users, n_items = ratings_matrix_train.shape
        print(f"Matrix dimensions: {n_users} users x {n_items} movies")

        # Initialize matrices
        self._init_matrices(n_users, n_items)

        # Get indices of non-zero elements
        users, items = ratings_matrix_train.nonzero()
        total_ratings = len(users)
        print(f"Total ratings to train on: {total_ratings}")

        # Training loop
        for epoch in range(self.n_epochs):
            epoch_start_time = time.time()

            # Shuffle the order of training examples
            shuffle_indices = np.random.permutation(len(users))

            # Track total error for the epoch
            total_error = 0

            for idx in shuffle_indices:
                u, i = users[idx], items[idx]
                r = ratings_matrix_train[u, i]

                # Compute current prediction
                pred = (self.global_mean +
                        self.user_biases[u] +
                        self.item_biases[i] +
                        self.user_factors[u] @ self.item_factors[i])

                # Compute error
                error = r - pred
                total_error += error ** 2

                # Update biases and factors
                self.user_biases[u] += self.learning_rate * (error - self.regularization * self.user_biases[u])
                self.item_biases[i] += self.learning_rate * (error - self.regularization * self.item_biases[i])

                user_factors_update = (error * self.item_factors[i] -
                                       self.regularization * self.user_factors[u])
                item_factors_update = (error * self.user_factors[u] -
                                       self.regularization * self.item_factors[i])

                self.user_factors[u] += self.learning_rate * user_factors_update
                self.item_factors[i] += self.learning_rate * item_factors_update

            # Print epoch summary
            rmse = np.sqrt(total_error / total_ratings)
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1}/{self.n_epochs}: RMSE = {rmse:.4f}, Time = {epoch_time:.2f}s")

        # Final training summary
        total_training_time = time.time() - start_total_time
        print(f"\n--- Training Complete ---")
        print(f"Total training time: {total_training_time:.2f} seconds")

        return self

    def evaluate(self, ratings_matrix_test: csr_matrix,
                 include_unrated: bool = False):
        """
        Evaluate model performance on test sparse matrix.

        Args:
            ratings_matrix_test (csr_matrix): Sparse test ratings matrix
            include_unrated (bool): Whether to include predictions for unrated items

        Returns:
            Dict with performance metrics, actual ratings, and predictions
        """
        print("\n--- Model Evaluation ---")

        # Compute predictions for test set
        predictions = []
        actual_ratings = []

        # Get test set non-zero indices
        test_users, test_items = ratings_matrix_test.nonzero()

        for idx in range(len(test_users)):
            u, i = test_users[idx], test_items[idx]
            actual_rating = ratings_matrix_test[u, i]

            # Translate back to original IDs if needed
            orig_user_id = self.reverse_user_id_map.get(u, u)
            orig_movie_id = self.reverse_movie_id_map.get(i, i)

            try:
                pred = self.predict(orig_user_id, orig_movie_id)
                predictions.append(pred)
                actual_ratings.append(actual_rating)
            except Exception as e:
                print(f"Prediction error for user {u}, movie {i}: {e}")
                continue

        # Compute metrics
        rmse = np.sqrt(mean_squared_error(actual_ratings, predictions))
        mae = mean_absolute_error(actual_ratings, predictions)

        print("Performance Metrics:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Total Test Ratings: {len(actual_ratings)}")

        return {
            'RMSE': rmse,
            'MAE': mae,
            'Total Test Ratings': len(actual_ratings)
        }, actual_ratings, predictions

    def save_model(self, filepath: str = 'svd_recommender_model.joblib'):
        """Save the entire model state to a file."""
        try:
            if self.user_factors is None:
                raise ValueError("Model must be trained before saving")

            model_state = {
                'user_factors': self.user_factors,
                'item_factors': self.item_factors,
                'user_biases': self.user_biases,
                'item_biases': self.item_biases,
                'global_mean': self.global_mean,
                'user_id_map': self.user_id_map,
                'movie_id_map': self.movie_id_map,
                'n_factors': self.n_factors,
                'learning_rate': self.learning_rate,
                'regularization': self.regularization,
                'n_epochs': self.n_epochs
            }

            joblib.dump(model_state, filepath)
            print(f"Model successfully saved to {filepath}")

        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filepath: str = 'svd_recommender_model.joblib'):
        """Load a previously saved model state."""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"No model file found at {filepath}")

            model_state = joblib.load(filepath)

            # Restore model parameters
            self.user_factors = model_state['user_factors']
            self.item_factors = model_state['item_factors']
            self.user_biases = model_state['user_biases']
            self.item_biases = model_state['item_biases']
            self.global_mean = model_state['global_mean']

            # Restore mapping dictionaries
            self.user_id_map = model_state['user_id_map']
            self.movie_id_map = model_state['movie_id_map']

            # Recreate reverse mapping dictionaries
            self.reverse_user_id_map = {idx: user for user, idx in self.user_id_map.items()}
            self.reverse_movie_id_map = {idx: movie for movie, idx in self.movie_id_map.items()}

            print(f"Model successfully loaded from {filepath}")
            print(f"Loaded model details:")
            print(f"  - Latent Factors: {model_state['n_factors']}")
            print(f"  - Unique Users: {len(self.user_id_map)}")
            print(f"  - Unique Movies: {len(self.movie_id_map)}")

            return self

        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def _init_matrices(self, n_users: int, n_items: int):
        """Initialize model parameters with logging."""
        print(f"\nInitializing matrices with {self.n_factors} latent factors")
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating with error handling and logging."""
        if self.user_factors is None:
            print("Error: Model must be trained before making predictions")
            return None

        try:
            u_idx = self.user_id_map[user_id]
            m_idx = self.movie_id_map[movie_id]
        except KeyError:
            return self.global_mean

        prediction = (self.global_mean +
                      self.user_biases[u_idx] +
                      self.item_biases[m_idx] +
                      self.user_factors[u_idx] @ self.item_factors[m_idx])

        return prediction

    def recommend_items(self, user_id: int, n_items: int = 10,
                        exclude_rated: bool = True):
        """Recommend items with detailed logging."""
        if self.user_factors is None:
            print("Error: Model must be trained before making recommendations")
            return []

        try:
            u_idx = self.user_id_map[user_id]
        except KeyError:
            print(f"Warning: User {user_id} not found in training data")
            return []

        # Calculate predictions for all items
        user_vector = (self.global_mean +
                       self.user_biases[u_idx] +
                       self.user_factors[u_idx] @ self.item_factors.T)

        # Create array of (movie_id, prediction) tuples
        predictions = []
        for m_idx, pred in enumerate(user_vector):
            movie_id = self.reverse_movie_id_map[m_idx]
            predictions.append((movie_id, pred))

        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Optional: exclude already rated movies
        if exclude_rated:
            rated_movies = set(df_filtered[df_filtered['sparse_user_id'] == user_id]['sparse_movie_id'])
            predictions = [rec for rec in predictions if rec[0] not in rated_movies]

        return predictions

    def handle_new_user(self, user_ratings, n_items=10):
        """
        Incorporate new user ratings into recommendations without full retraining

        Args:
            user_ratings (pd.DataFrame): DataFrame with columns 'movieId' and 'rating'
            n_items (int): Number of recommendations to return

        Returns:
            List of recommended movie IDs
        """
        # Check if model is trained
        if self.user_factors is None:
            raise ValueError("Model must be trained first")

        # Validate movie IDs exist in original training data
        valid_movies = [mid for mid in user_ratings['sparse_movie_id'] if mid in self.movie_id_map]
        valid_ratings = user_ratings[user_ratings['sparse_movie_id'].isin(valid_movies)]

        if len(valid_movies) == 0:
            print("No valid movie ratings found")
            return []

        # Ensure consistent dimensionality
        new_user_factors = np.zeros(self.item_factors.shape[1])
        new_user_bias = 0

        for _, row in valid_ratings.iterrows():
            movie_idx = self.movie_id_map[row['sparse_movie_id']]

            # Safely handle potential shape mismatches
            movie_factors = self.item_factors[movie_idx]

            # Ensure correct dimensionality
            if len(movie_factors) != len(new_user_factors):
                # Truncate or pad the movie factors to match new_user_factors
                if len(movie_factors) > len(new_user_factors):
                    movie_factors = movie_factors[:len(new_user_factors)]
                else:
                    padded_factors = np.zeros(len(new_user_factors))
                    padded_factors[:len(movie_factors)] = movie_factors
                    movie_factors = padded_factors

            # Update user vector with weighted item factors
            new_user_factors += row['rating'] * movie_factors
            new_user_bias += row['rating'] - self.global_mean

        # Normalize by number of ratings
        new_user_factors /= len(valid_ratings)
        new_user_bias /= len(valid_ratings)

        # Compute predictions for all items
        predictions = (self.global_mean +
                       new_user_bias +
                       new_user_factors @ self.item_factors.T)

        # Create and sort recommendations
        recommendations = [
            (self.reverse_movie_id_map[idx], pred)
            for idx, pred in enumerate(predictions)
        ]
        recommendations.sort(key=lambda x: x[1], reverse=True)

        # Exclude movies already rated
        rated_movies = set(valid_ratings['sparse_movie_id'])
        recommendations = [rec for rec in recommendations if rec[0] not in rated_movies]

        return recommendations[:n_items]








