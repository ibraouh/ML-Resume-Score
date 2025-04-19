from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI(title="Resume–Job Match Scorer")

# NEEDS TO BE CHANGED TO DOMAIN IN THE FUTURE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # or ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST, OPTIONS, etc.
    allow_headers=["*"],  # Allow any headers
)


class MatchRequest(BaseModel):
    resume_text: str
    keywords: list[str]


class MatchResponse(BaseModel):
    match_score: float  # percentage between 0 and 100


@app.post("/score", response_model=MatchResponse)
def score_match(req: MatchRequest):
    if not req.resume_text or not req.keywords:
        raise HTTPException(
            status_code=400, detail="Both resume_text and keywords are required."
        )
    print(f"Received request: {req}")
    # print(f"Resume text: {req.resume_text}")
    # print(f"Keywords: {req.keywords}")

    # 1. Prepare documents
    kw_doc = " ".join(req.keywords)
    docs = [req.resume_text, kw_doc]

    # 2. TF–IDF vectorization
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(docs)  # shape: (2, n_features)

    # 3. Cosine similarity between resume (0) and keywords (1)
    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0, 0]

    # 4. Convert to percentage
    percentage = float(np.round(sim * 100, 2))

    return MatchResponse(match_score=percentage)
