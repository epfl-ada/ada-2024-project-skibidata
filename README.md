# A Movie Recommendation System
## Abstract
Recommendation systems are everywhere ...
A 150 word description of the project idea and goals. What’s the motivation behind your project? What story would you like to tell, and why?
## Research Questions
1. ...
2. How to compare the results obtained by different recommandation systems ?
   
## Additional datasets
- __Movielens Dataset:__
   - Description:
     
     >This dataset describes 5-star rating and free-text tagging activity from MovieLens, a movie recommendation service. It contains 32000204 ratings and 2000072 tag applications across 87585 movies. These data were created by 200948 users between January 09, 1995 and October 12, 2023. This dataset was generated on October 13, 2023. Users were selected at random for inclusion. All selected users had rated at least 20 movies.
   - Usage: Get reviews for recommandation system
   - Available at: https://grouplens.org/datasets/movielens/
- __TMdB Dataset:__
   - Description:
     
     > The TMDb (The Movie Database) is a comprehensive movie database that provides information about movies, including details like titles, ratings, release dates, revenue, genres, and much more. This dataset contains a collection of 1,000,000 movies from the TMDB database.
   - Usage: Complete and correct the CMU Dataset
   - Available at : https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies/data
- __IMdB Infos:__
   - PYthon library to get infos
   - IMdB API
  
List the additional dataset(s) you want to use (if any), and some ideas on how you expect to get, manage, process, and enrich it/them. Show us that you’ve read the docs and some examples, and that you have a clear idea on what to expect. Discuss data size and format if relevant. It is your responsibility to check that what you propose is feasible.
## Methods
There is mostly three types of recommandation system: 
- Content-based filtering:

  > Content-based filtering is a recommendation strategy that suggests items similar to those a user has previously liked. It calculates similarity (often using cosine similarity) between the user’s preferences and item attributes, such as lead actors, directors, and genres. For example, if a user enjoys ‘The Prestige,’ the system recommends movies with Christian Bale, the ‘Thriller’ genre, or films by Christopher Nolan. However, content-based filtering has drawbacks. It limits exposure to different products, preventing users from exploring a variety of items. This can hinder business expansion as users might not try out new types of products.

- Collaborative filtering:

  > Collaborative filtering is a recommendation strategy that considers the user’s behavior and compares it with other users in the database. It uses the history of all users to influence the recommendation algorithm. Unlike a content-based recommender system, a collaborative filtering recommender relies on multiple users’ interactions with items to generate suggestions. It doesn’t solely depend on one user’s data for modeling. There are various approaches to implementing collaborative filtering, but the fundamental concept is the collective influence of multiple users on the recommendation outcome. There are 2 types of collaborative filtering algorithms....

- Demographic filtering:

  > demographic filtering....


- __Task 1: Cleaning and Exploratory Data Analysis of CMU Dataset__
   - __1.1__: Clean CMU dataset
   - __1.2__:Complete and correct CMU dataset
   - __1.3__: EDA on the new CMU dataset
- __Task 2: First Recommendation System: Demographic Filtering__
   - __2.1__: Cleaing and first data analysis on Movielens dataset
   - __2.1__: Implement demographic filtering to obatin first results
- __Task 3: Improving the Recommendation System: Content-based and Collaborative Filtering__      
   - __3.1__: Implement content-based filtering      
   - __3.2__: Implement collaborative filtering 
- __Task 4: An even better Recommendation Sytem ? Hybrid Filtering__
   - __4.1__: Implement an hybrid recommendation system with both content-based and collaborative filtering
## Proposed timeline
- [x] 
- [x] 15 Nov. 2024 : Milestone P2
- [ ] TO DO
- [ ] TO DO 
- [ ] 20 Dec. 2024 : Milestone P3
## Organization within the team
- Vincent: Free loader
- Mayeul:
- Arthur: Follower
- Alex: Team leader
- Corentin:
## Organization of the Github
- results.ipynb : notebook that integrates all steps of the project, including data preprocessing, exploratory data visualization, feature engineering, and model building.
- Draft : folder containing files used to store experimental concepts and test scripts
- README.md : 
-.gitignore : Configuration file used to specify files and directories that should be ignored by Git. 
## Questions for TAs (optional)
Add here any questions you have for us related to the proposed project.
