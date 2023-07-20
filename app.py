from flask import Flask, render_template, request
from recommender import recommenderSystem as rs

app = Flask(__name__)

recommender = rs('movies.csv')
recommender.build_model()

@app.route('/')
def index():
    movie_titles = recommender.movies['title'].tolist()

    return render_template('index.html', movie_titles=movie_titles)


@app.route('/recommend', methods=['POST'])
def recommend():
    movie_titles = request.form['movie_name']
    top_movie = int(request.form['top_movie'])

    scores, movie_indices = recommender.genre_recommendations(movie_titles, top_movie)
    title_movie = recommender.movies['title'].iloc[movie_indices].values
    overview_movie = recommender.movies['overview'].iloc[movie_indices].values

    recommended_movies = zip(title_movie,overview_movie)

    return render_template('recommend.html', recommended_movies=recommended_movies , movie_name=movie_titles)


if __name__ == '__main__':
    app.run(debug=True)