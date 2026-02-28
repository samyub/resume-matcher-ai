from sklearn.feature_extraction.text import TfidfVectorizer

def create_vectorizer():
    return TfidfVectorizer()

def vectorize_text(vectorizer, text_list):
    return vectorizer.fit_transform(text_list)
