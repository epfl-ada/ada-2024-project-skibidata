# A Movie Recommendation System
## Abstract
Recommendation systems are everywhere in today's digital world, enhancing user experiences across streaming platforms, e-commerce websites, and social media. This project aims to develop a movie recommendation system using a publicly available movie dataset. The goal is to analyze user preferences, movie genres, and ratings to predict and suggest films that align with a user's unique tastes. By employing techniques such as collaborative filtering, content-based filtering, and hybrid approaches, the system will provide personalized movie recommendations.

The motivation behind this project comes from the growing demand for intelligent systems that can navigate vast amounts of content, ensuring users find what they love. Additionally, this project showcases the power of data-driven decision-making and machine learning in solving real-world problems, with potential applications in entertainment and beyond.
Ca vient de ChatGPT ça mérite d'être retravaillé
## Research Questions
1. ...
2. How to compare the results obtained by different recommandation systems ?
   
## Additional datasets
- __TMdN and Movielens Dataset:__
   - Description: We use a dataset available on [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset). This dataset contains features from both [TMdB](https://www.themoviedb.org/) and [Movielens](https://grouplens.org/datasets/movielens/) websites. The TMDb (The Movie Database) is a comprehensive movie database that provides information about movies, including details like titles, ratings, release dates, revenue, genres. Movienlens is a movie recommendation system. Users can rate movies on a scale from 1 to 5.
   - Usage: We will use the TMdB features to both correct and complete the CMU dataset. Indeed, the CMU dataset contains a lot of wrong information. Furthermore, it might be useful to have some additional features such as the director of the movies. The ratings from the Movielens dataset are going to be used when building a recommendation system. We might loose some of the movies from the original dataset by doing so. However, we think it is better to have a cleaner of the dataset even if it means to only take a subset of the original one.
- __IMdB:__
   - We might use (not done yet) the API from [IMdB](https://www.imdb.com/) to retrieve missing values when possible.
  
## Methods
There is mostly three types of recommandation system: 
- Content-based filtering:
  > Content-based filtering is a recommendation strategy that suggests items similar to those a user has previously liked. It calculates similarity (often using cosine similarity) between the user’s preferences and item attributes, such as lead actors, directors, and genres. For example, if a user enjoys ‘The Prestige,’ the system recommends movies with Christian Bale, the ‘Thriller’ genre, or films by Christopher Nolan. However, content-based filtering has drawbacks. It limits exposure to different products, preventing users from exploring a variety of items. This can hinder business expansion as users might not try out new types of products.

- Collaborative filtering:

  > Collaborative filtering is a recommendation strategy that considers the user’s behavior and compares it with other users in the database. It uses the history of all users to influence the recommendation algorithm. Unlike a content-based recommender system, a collaborative filtering recommender relies on multiple users’ interactions with items to generate suggestions. It doesn’t solely depend on one user’s data for modeling. There are various approaches to implementing collaborative filtering, but the fundamental concept is the collective influence of multiple users on the recommendation outcome. There are 2 types of collaborative filtering algorithms....

- Demographic filtering:

  > demographic filtering....

We will implement all three of these recommendation systems. We will compare their recommendation 


- __Task 1: Cleaning and Exploratory Data Analysis of CMU Dataset__
   - __1.1__: Clean, correct and complete CMU dataset 
   - __1.2__: Perform some exploratory data analysis on the new dataset
- __Task 2: First Recommendation System: Demographic Filtering__
   - __2.1__: Implement demographic filtering
- __Task 3: Improving the Recommendation System: Content-based and Collaborative Filtering__      
   - __3.1__: Implement content-based filtering (find ways to cluster movies by similarities)         
   - __3.2__: Implement collaborative filtering (find ways to cluster users by same taste)  
- __Task 4: An even better Recommendation Sytem ? Hybrid Filtering__
   - __4.1__: Implement an hybrid recommendation system with both content-based and collaborative filtering
     
## Proposed timeline 
- [x] 15 Nov. 2024: Milestone P2
- [ ] 22 Nov. 2024: (Optional) Retreive more missing values from IMdB
- [ ] 22 Nov. 2024: Finish the exploratory data analysis with analysis more specified to our goal
- [ ] 06 Dec. 2024: Finish all recommendation systems (demographic, content-based and collaborative)
- [ ] 20 Dec. 2024 : Milestone P3
## Organization within the team
- Vincent: will work on collaborative filtering
- Mayeul: will work on collaborative filtering
- Arthur: will work on the final website integrating our different recommendation systems 
- Alex: will work on content-based filtering
- Corentin: will work on content-based filtering
## Organization of the Github
- results.ipynb: notebook that integrates all steps of the project, including data preprocessing, exploratory data visualization, feature engineering, and model building.
- Draft: folder containing files used to store experimental concepts and test scripts
- Data: folder containing some csv files that contains some queried or downloaded data needed for different part of the results notebook
- README.md
-.gitignore : Configuration file used to specify files and directories that should be ignored by Git. 

