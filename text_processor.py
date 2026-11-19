import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TextProcessor:
    def __init__(self):
        self.stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with'}
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.venue_names = []
