import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TextProcessor:
    """Cleans venue descriptions and builds TF-IDF features for Track A NLP."""
    def __init__(self):
        self.stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with'}
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.venue_names = []

    def clean(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        filtered = [t for t in tokens if t not in self.stopwords]
        return " ".join(filtered)

    def normalise(self, texts):
        return [self.clean(t) for t in texts]

    def fit_tfidf(self, venue_names, texts):
        self.venue_names = venue_names
        cleaned_texts = self.normalise(texts)
        self.tfidf_matrix = self.vectorizer.fit_transform(cleaned_texts)

    def similar_venues(self, target_venue, top_n=3):
        if target_venue not in self.venue_names:
            return []
        idx = self.venue_names.index(target_venue)
        sim_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        similar_indices = sim_scores.argsort()[::-1][1:top_n+1]
        return [(self.venue_names[i], sim_scores[i]) for i in similar_indices]