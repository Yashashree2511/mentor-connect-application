from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

# initialize app
app = FastAPI()

# -----------------------------
# Request Schema (for POST API)
# -----------------------------
class QueryRequest(BaseModel):
    query: str
    min_experience: int = 0
    domain: str = None


# -----------------------------
# Load model and data (once)
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

index = faiss.read_index("faiss_index.bin")

with open("id_mapping.pkl", "rb") as f:
    id_mapping = pickle.load(f)

df = pd.read_csv("cleaned_data.csv")


# -----------------------------
# Recommendation API
# -----------------------------
@app.post("/recommend")
def recommend(request: QueryRequest):

    query = request.query
    min_experience = request.min_experience
    domain = request.domain

    # convert query to embedding
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    # search FAISS
    D, I = index.search(query_embedding, 5)

    mentor_ids = [id_mapping[i] for i in I[0]]
    similarities = D[0]

    # fetch data
    results = df.set_index('mentor_id').loc[mentor_ids].reset_index()
    
    # scoring
    results['similarity'] = similarities
    results['final_score'] = 0.7 * results['similarity'] + 0.3 * (results['rating'] / 5)

    # filters
    results = results[results['experience_years'] >= min_experience]

    if domain:
        results = results[results['domain'].str.contains(domain, case=False)]

    # sort + top 5
    results = results.sort_values(by='final_score', ascending=False).head(5)

    # clean response
    return results[
        [
            'mentor_id',
            'name',
            'skills',
            'experience_years',
            'rating',
            'similarity',
            'final_score'
        ]
    ].to_dict(orient="records")