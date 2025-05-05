from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

app = FastAPI(title="Resume-Job Match Scorer")

# NEEDS TO BE CHANGED TO DOMAIN IN THE FUTURE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] to allow all origins
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
def score(req: MatchRequest):
    if not req.resume_text or not req.keywords:
        raise HTTPException(
            status_code=400, detail="Both resume_text and keywords are required."
        )
    print(f"Received request: {req}")

    pct = score_match(req.resume_text, req.keywords)
    print(f"Match percentage: {pct:.2f}%")

    return MatchResponse(match_score=pct)


try:
    from sentence_transformers import SentenceTransformer

    _sbert = SentenceTransformer("all-MiniLM-L6-v2")
except ImportError:
    _sbert = None  # fall back if not installed


def keyword_overlap_score(resume: str, keywords: list[str]) -> float:
    """Fraction of keywords found in the resume."""
    matched = 0
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}\b", resume, re.IGNORECASE):
            matched += 1
    return matched / max(len(keywords), 1)


def tfidf_semantic_score(resume: str, keywords: list[str]) -> float:
    """Cosine similarity between resume & keywords doc via TF–IDF."""
    docs = [resume, " ".join(keywords)]
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    mat = vect.fit_transform(docs)
    return float(cosine_similarity(mat[0:1], mat[1:2])[0, 0])


def embedding_score(resume: str, keywords: list[str]) -> float:
    """Cosine similarity in SBERT embedding space. Returns 0 if SBERT unavailable."""
    if _sbert is None:
        return 0.0
    emb_resume = _sbert.encode(resume, convert_to_numpy=True, normalize_embeddings=True)
    emb_kw = _sbert.encode(
        " ".join(keywords), convert_to_numpy=True, normalize_embeddings=True
    )
    return float(cosine_similarity([emb_resume], [emb_kw])[0, 0])


def score_match(
    resume: str,
    keywords: list[str],
    w_overlap: float = 0.4,
    w_tfidf: float = 0.2,
    w_embed: float = 0.4,
) -> float:
    """
    Ensemble score in [0,100]:
      - w_overlap: weight for exact keyword matches
      - w_tfidf:   weight for TF-IDF semantic similarity
      - w_embed:   weight for embedding similarity
    """
    # 1) compute sub‑scores
    s1 = keyword_overlap_score(resume, keywords)
    s2 = tfidf_semantic_score(resume, keywords)
    s3 = embedding_score(resume, keywords)

    # 2) normalize weights to sum=1
    total = w_overlap + w_tfidf + w_embed
    w1, w2, w3 = w_overlap / total, w_tfidf / total, w_embed / total

    # 3) weighted combination
    combined = w1 * s1 + w2 * s2 + w3 * s3

    # 4) as percentile
    pct = combined * 100.0
    for threshold, delta in [
        (70, 20),
        (67, 12),
        (65, 5),
    ]:
        if pct >= threshold:
            print(f"Editing {pct:.2f}% to {pct + delta:.2f}%")
            pct = pct + delta
            break
    pct = min(pct, 100.0)
    return pct
