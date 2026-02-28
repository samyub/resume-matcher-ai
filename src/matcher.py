from sklearn.metrics.pairwise import cosine_similarity
from .preprocessing import clean_text
from .vectorizer import create_vectorizer

def calculate_match_score(resume_text, job_text):
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_text)

    vectorizer = create_vectorizer()
    vectors = vectorizer.fit_transform([resume_clean, job_clean])

    similarity = cosine_similarity(vectors[0], vectors[1])
    return float(similarity[0][0])
