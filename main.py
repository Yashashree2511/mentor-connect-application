from fastapi import FastAPI
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

app = FastAPI()

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("faiss_index.bin")

with open("id_mapping.pkl", "rb") as f:
    id_mapping = pickle.load(f)

df = pd.read_csv("cleaned_data_v2.csv")


@app.get("/recommend")   
def recommend(query: str):

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    D, I = index.search(query_embedding, 5)

    mentor_ids = [id_mapping[i] for i in I[0]]
    similarities = D[0]

    # get data
    results = df[df['mentor_id'].isin(mentor_ids)].copy()

    # add similarity
    results['similarity'] = similarities

    # compute final score
    results['final_score'] = 0.7 * results['similarity'] + 0.3 * (results['rating'] / 5)

    # sort
    results = results.sort_values(by='final_score', ascending=False)

    return results.to_dict(orient="records")