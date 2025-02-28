# Book-Recommendation-using-KNN
This project implements a **K-Nearest Neighbors (KNN) based book recommendation system**. The model suggests books similar to a given book based on user ratings and features.

## Features
- **Collaborative Filtering** – Recommends books based on user preferences.
- **KNN Algorithm** – Finds books with similar ratings and features.
- **Dataset Processing** – Cleans and preprocesses book rating datasets.
- **Scalability** – Efficient for large book datasets.
- **Interactive Usage** – Users can input book titles to receive recommendations.

## Code Samples
python
# import libraries (you may add additional imports but you may not have to)
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

python
# get data files
!wget https://cdn.freecodecamp.org/project-data/books/book-crossings.zip

!unzip book-crossings.zip

books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'

python
# import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})


## Installation & Usage
1. Clone the repository:
   bash
   git clone https://github.com/yourusername/book-recommendation-knn.git
   
2. Install dependencies:
   bash
   pip install -r requirements.txt
   
3. Run the recommendation system:
   bash
   python recommend.py --book "The Great Gatsby"
