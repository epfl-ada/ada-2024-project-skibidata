# A Movie Recommendation System
## Abstract
Recommendation systems are everywhere in today's digital world, enhancing user experiences across streaming platforms, e-commerce websites, and social media. This project aims to develop a movie recommendation system using a publicly available movie dataset. The goal is to analyse user preferences, movie genres, and ratings to predict and suggest films that align with a user's unique tastes. By employing techniques such as collaborative filtering, content-based filtering, and hybrid approaches, the system will provide personalized movie recommendations.

The motivation behind this project comes from the growing demand for intelligent systems that can navigate vast amounts of content, ensuring users find what they love. Additionally, this project showcases the power of data-driven decision-making and machine learning in solving real-world problems, with potential applications in entertainment and beyond.

## Research Questions
1. How to optimally cluster movies by similarities for a recommendation system ?
2. How to optimally cluster users by tastes for a recommendation system ?
3. How to compare the results obtained by different recommandation systems ?
   
## Additional datasets
- __TMdN and Movielens Dataset:__
  
   - _Description_:
     
     We use a dataset available on [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset). This dataset contains features from both [TMdB](https://www.themoviedb.org/) and [Movielens](https://grouplens.org/datasets/movielens/) websites. The TMDb (The Movie Database) is a comprehensive movie database that provides information about movies, including details like titles, ratings, release dates, revenue, genres. Movienlens is a movie recommendation system. Users can rate movies on a scale from 1 to 5.
     
   - _Usage_:
     
     We will use the TMdB features to both correct and complete the CMU dataset. Indeed, the CMU dataset contains a lot of wrong information. Furthermore, it might be useful to have some additional features such as the director of the movies. The ratings from the Movielens dataset are going to be used when building a recommendation system. We might loose some of the movies from the original dataset by doing so. However, we think it is better to have a cleaner of the dataset even if it means to only take a subset of the original one.
     
- __IMdB:__
  
   - _Usage_:
     
     We might use (not done yet) the API from [IMdB](https://www.imdb.com/) to retrieve missing values when possible.
  
## Methods
We will consider two types of recommandation system: 
- Content-based filtering:
  
  > Content-based filtering is a recommendation strategy that suggests items similar to those a user has previously liked. It calculates similarity between the user’s preferences and item attributes, such as lead actors, directors, and genres. However, content-based filtering has drawbacks. It limits exposure to different products, preventing users from exploring a variety of items.

- Collaborative filtering:

  > Collaborative filtering is a recommendation strategy that considers the user’s behavior and compares it with other users in the database. It uses the history of all users to influence the recommendation algorithm. Unlike a content-based recommender system, a collaborative filtering recommender relies on multiple users’ interactions with items to generate suggestions. It doesn’t solely depend on one user’s data for modeling. There are various approaches to implementing collaborative filtering, but the fundamental concept is the collective influence of multiple users on the recommendation outcome.

We will implement the two systems. We will compare their recommendation based on metrics that still need to be determined and discussed. 


- __Task 1: Cleaning and Exploratory Data Analysis of CMU Dataset__
   - __1.1__: Clean, correct and complete CMU dataset 
   - __1.2__: Perform some exploratory data analysis on the new dataset
- __Task 2: Improving the Recommendation System: Content-based and Collaborative Filtering__      
   - __2.1__: Implement content-based filtering (find ways to cluster movies by similarities)         
   - __2.2__: Implement collaborative filtering (find ways to cluster users by same taste)  
- __Task 3: An even better Recommendation Sytem ? Hybrid Filtering__
   - __3.1__: Implement an hybrid recommendation system with both content-based and collaborative filtering
- __Task 4: A live demo__
   - __4.1__: Integrate the recommendation systems as a live demo in the website as well as wrinting the data story 
     
## Proposed timeline 
- [x] 15 Nov. 2024: Milestone P2
- [x] 15 Nov. 2024: __Task 1.1__
- [ ] 22 Nov. 2024: (Optional) Retreive more missing values from IMdB
- [ ] 22 Nov. 2024: Finish the exploratory data analysis with analysis more specific (__Task 1.2__)
- [ ] 06 Dec. 2024: Finish all recommendation systems (content-based and collaborative) (__Task 2__ and __Task 3__)
- [ ] 20 Dec. 2024 : Milestone P3: integrate all recommendation systems to have a demo on the website as well as writing the data story (__Task 4__)

## Organization within the team
Actually, the following members are working on:
- Vincent: collaborative filtering
- Mayeul: collaborative filtering
- Arthur: the final website integrating our different recommendation systems
- Alex: content-based filtering
- Corentin: finishing the cleaning and exploratory data analysis as well as working on the hybrid recommendation system
  
## Organization of the Github
- results.ipynb: notebook that integrates all steps of the project, including data preprocessing, exploratory data visualization, feature engineering, and model building.
- Draft: folder containing files used to store experimental concepts and test scripts
- Data: folder containing some csv files that contains some queried or downloaded data needed for different part of the results notebook
- README.md
-.gitignore : Configuration file used to specify files and directories that should be ignored by Git. 

