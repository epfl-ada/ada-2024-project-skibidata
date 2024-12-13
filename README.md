# A Movie Recommendation System
## Abstract
Recommendation systems are everywhere in today's digital world, enhancing user experiences across streaming platforms, e-commerce websites, and social media. This project aims to develop a movie recommendation system using a publicly available movie dataset. The goal is to analyse user preferences, movie genres, and ratings to predict and suggest films that align with a user's unique tastes. By employing techniques such as collaborative filtering, content-based filtering, and hybrid approaches, the system will provide personalized movie recommendations.

The motivation behind this project comes from the growing demand for intelligent systems that can navigate vast amounts of content, ensuring users find what they love. Additionally, this project showcases the power of data-driven decision-making and machine learning in solving real-world problems, with potential applications in entertainment and beyond.

## Research Questions
1. How effectively can collaborative filtering and content-based filtering predict user preferences in movie recommendations ?
   - 1.1 What are the most important features when trying to cluster movies by similarities ?
   - 1.2 Similarly, how to find users with the same preferences ?
   - 1.3 How to compare the results obtained by different recommendation systems ?
   - 1.4 What are the key differences in performance between user-based and item-based collaborative filtering approaches ?
2. Can an hybrid recommendation system outperfom collaborative filtering and/or content-based filtering approaches ?
   - 2.1 What weighting or blending strategies work best for hybrid systems ?  
   
## Additional datasets
- __TMdB and Movielens Dataset:__
  
   - _Description_:
     
     We use a dataset available on [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset). This dataset contains features from both [TMdB](https://www.themoviedb.org/) and [Movielens](https://grouplens.org/datasets/movielens/) websites. The TMDb (The Movie Database) is a comprehensive movie database that provides information about movies, including details like titles, ratings, release dates, revenue, genres. Movienlens is a movie recommendation system. Users can rate movies on a scale from 1 to 5.
     
   - _Usage_:
     
     We will use the TMdB features to both correct and complete the CMU dataset. Indeed, the CMU dataset contains a lot of wrong information. Furthermore, it might be useful to have some additional features such as the director of the movies. The ratings from the Movielens dataset are going to be used when building a recommendation system. We might loose some of the movies from the original dataset by doing so. However, we think it is better to have a cleaner dataset even if it means to only take a subset of the original one.
     
- __IMdB:__

  - _Description_:
 
    IMdB is an online database of information related to films, television series, podcasts, home videos, video games, and streaming content online – including cast, production crew and personal biographies, plot summaries, trivia, ratings, and fan and critical reviews.
  
   - _Usage_:
     
     We use the API from [IMdB](https://www.imdb.com/) to retrieve basic informations
  
## Methods
We will consider two types of recommandation system: 
- Content-based filtering:
  
  > Content-based filtering is a recommendation strategy that suggests items similar to those a user has previously liked. It calculates similarity between the user’s preferences and item attributes, such as lead actors, directors, and genres. However, content-based filtering has drawbacks. It limits exposure to different products, preventing users from exploring a variety of items.

- Collaborative filtering:

  > Collaborative filtering is a recommendation strategy that considers the user’s behavior and compares it with other users in the database. It uses the history of all users to influence the recommendation algorithm. Unlike a content-based recommender system, a collaborative filtering recommender relies on multiple users’ interactions with items to generate suggestions. It doesn’t solely depend on one user’s data for modeling. There are various approaches to implementing collaborative filtering, but the fundamental concept is the collective influence of multiple users on the recommendation outcome.

For each of these types of filtering we implement different recommender systems:
- Content-based filtering:
   - Autoencoder
   - ...
- Collaborative filtering:
   - User-User similarity
   - Item-Item similarity
   - SVD

The recommender systems are then compared with metrics such as RMSE or MAE when possible. Otherwise we will compare the order of recommendations with the actual ratings given by a user.  

## Contribution of each member to the project
- _Vincent_ : collaborative filtering
- _Mayeul_ : some visualization
- _Arthur_ : the final website that will integrate our different recommendation systems
- _Alex_ : content-based filtering
- _Corentin_ : finishing the cleaning and exploratory data analysis as well as working on the hybrid recommendation system
  
## Organization of the Github
Our Github is organized as follows:
- `results.ipynb` notebook that integrates all steps of the project, including data preprocessing, exploratory data analysis and some vizualization, feature engineering, and first model building.
- `helpers.py` python file containing all functions needed for visualization and cleaning.
- `collaborative.py` python file containing all functions needed for the collaborative filtering based recommender systems.
- `content_based.py` python file containing all functions needed for the content-based filtering based recommender systems.
- `Draft` folder containing files used to store experimental concepts and test scripts. Nothing relevant for P2 deliverable can be found in this folder. 
- `Data` folder containing some csv files that contains some queried or downloaded data needed for different part of the results notebook. 
- `README.md`
- `requirements.txt` text file specifying the necessary libraries to run the results noetbook. 
- `gitignore` configuration file used to specify files and directories that should be ignored by Git.
  
In order to run the current `results.ipynb` notebook you need to download the dataset presented above from [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) and store it in a folder called _Data/TMdB/_ as well as download the [CMU Movie Summary Corpus Dataset](http://www.cs.cmu.edu/~ark/personas/) and store it in _Data/MovieSummaries/_.
