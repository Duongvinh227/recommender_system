import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class recommenderSystem(object):
    def __init__(self, path_csv):
        self.movies = self.get_data_frame(path_csv)
        self.matrix = None
        self.consine = None

    def build_model(self):
        self.movies['genres'] = self.movies['genres'].str.split('|')
        self.movies['genres'] = self.movies['genres'].fillna("").astype('str')

        self.matrix = self.handle_tfidf_matrix()
        self.consine = self.handel_cosine_sim()

    def get_data_frame(self, path_csv):
        movie_cols = ['id', 'title', 'genres', 'overview']
        movies_full = pd.read_csv(path_csv, usecols=movie_cols, encoding='latin-1')
        movies = movies_full.head(1000)
        movies_clean = movies.dropna()

        return movies_clean

    def handle_tfidf_matrix(self):
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1))
        new_tfidf_matrix = tf.fit_transform(self.movies['genres'])
        return new_tfidf_matrix

    def handel_cosine_sim(self):
        new_cosine_sim = linear_kernel(self.matrix, self.matrix)
        return new_cosine_sim

    def genre_recommendations(self, movie_name, top_movie):
        indices = pd.Series(self.movies.index, index=self.movies['title'])
        idx = indices[movie_name]
        sim_scores = list(enumerate(self.consine[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_movie + 1]
        movie_indices = [i[0] for i in sim_scores]
        return sim_scores, movie_indices
