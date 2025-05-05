from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re


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
    w_overlap: float = 0.6,
    w_tfidf: float = 0.2,
    w_embed: float = 0.2,
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
        # (85, 25),
        # (80, 20),
        (70, 20),
        (67, 12),
        (65, 5),
        # (60, 5),
    ]:
        if pct >= threshold:
            print(f"Editing {pct:.2f}% to {pct + delta:.2f}%")
            pct = pct + delta
            break
    pct = min(pct, 100.0)
    return pct


def score(resume, keywords):
    pct = score_match(resume, keywords)
    print(f"Score: {pct:.2f}%")
    return pct


resume = """
Ibrahim (Abe) Raouh
New York, NY • (332) 207-5303 • abe@raouh.com • linkedin.com/in/iraouh • github.com/ibraouh • raouh.com
EDUCATION
Fordham University, Graduate School of Arts and Sciences – New York, NY Master of Science in Computer Science, Concentration in Artificial Intelligence | GPA: 3.7/4.0
Relevant coursework: Distributed Systems, Data Science Math, Data Mining, iOS, AI, ML, Database Systems, Algorithms
Fordham University, College of Rose Hill – Bronx, NY Bachelor of Science in Math and Computer Science | Magna Cum Laude
Relevant coursework: C++, Algorithms, OS, Data Structures, Web Development, Software Engineering, Networking
May 2025
May 2024
SKILLS
●
●
WORK EXPERIENCE
Languages: Python (3yrs), C++ (5yrs), C (6yrs), JavaScript (2yrs), TS (1yr), Swift (2yrs), SQL (1yr), PHP (1yr), Rust (1yr)
Tools: AWS, Google Cloud, Docker, MySQL, Git, Kubernetes, TensorFlow, Jupyter Notebook, ROS, Bash, New Relic, Solace
Junior Software Engineer - MBO Partners | New York, NY January 2024 – Present
●
Designed and implemented an automatic, thread-safe, and lightweight contextual logging system to reduce debugging
time from 74 minutes to seconds, enhancing observability and issue resolution in production
●
Led the integration of the automated logging enhancements into a high-volume invoicing system, improving error
tracking, debugging efficiency, and system reliability
●
Working on a critical API integration between MBO and PwC that facilitates $40 million in revenue, ensuring seamless
and secure data synchronization. Implementing Solace’s messaging system in the backend to efficiently transmit real-time
updates to PwC’s secure system, improving system interoperability and reliability
Software Engineering Intern - Mobiblanc | Casablanca, Morocco Demo | June 2022 – August 2022
●
Developed a modular natural language processing (NLP) customer support bot using DialogFlow on Google Cloud and
implemented it on 7 web pages using NodeJS. The bot decreased wait time by over 70% and reduced costs by 45%
RESEARCH EXPERIENCE
Graduate Research Assistant - Professor Wen Li, Fordham | New York, NY Github | June 2023 – September 2023
●
Implemented the iterative thresholding method for image segmentation using the Chan-Vese model in MATLAB and
Python; incorporated convolution operations and region-specific information to enhance accuracy and achieve a 40%
reduction in false positives
Undergraduate Research Assistant - Professor Ying Mao, Fordham | New York, NY April 2022 – April 2023
●
Developed a multi-tenant quantum simulator for deep learning application on a classical computer with Python and
TensorFlow. Employed Qiskit APIs to analyze the workload distribution across individual qubits, gaining valuable
efficiency in performance and capability, which reduces development time by over 3 hours
●
Utilized Docker, Git, and Agile Scrum process to facilitate and accelerate the development process within the team
PERSONAL PROJECTS
ML Job Tracker | Full Stack Web Development and Machine Learning View Project | GitHub
●
Used React, Next.js, Firebase, Tailwind CSS, and ChatGPT API to build a comprehensive job search management platform
●
Implemented real-time data synchronization, server-side rendering, and AI-powered job parsing to create a responsive
and intuitive interface for tracking multiple job applications, visualizing progress, and automating data entry
●
Implemented a custom ML algorithm that analyzes similarities between a resume and a job posting to suggest edits
NFL Big Data Bowl 2025 (Kaggle Competition) | Data Science and Machine Learning Kaggle | GitHub
●
Developed a Random Forest classifier to predict Football play types (Run vs. Pass) in NFL games using real-time player
tracking data, achieving 86% training accuracy and 79% testing accuracy
●
Engineered position-based distance metrics and player movement features to enhance model performance, leveraging
feature engineering and data preprocessing techniques
●
Applied Python, scikit-learn, and statistical analysis to derive insights from large-scale datasets, demonstrating
data-driven decision-making and machine learning expertise
"""
kw = [
    "Python",
    "C++",
    "JavaScript",
    "Machine Learning",
    "Data Science",
    "Software Engineering",
    "Algorithms",
    "Distributed Systems" "Natural Language Processing",
    "NLP",
    "TensorFlow",
    "Google Cloud",
    "AWS",
    "Docker",
    "MySQL",
    "Git",
    "Kubernetes",
    "Jupyter Notebook",
    "ROS",
    "Bash",
    "New Relic",
    "Solace",
    "React",
    "Next.js",
    "Firebase",
    "Tailwind CSS",
    "ChatGPT API",
    "NodeJS",
    "DialogFlow",
    "Qiskit",
    "MATLAB",
    "Agile Scrum",
    "Random Forest",
    "Feature Engineering",
    "Statistical Analysis",
    "Server-Side Rendering",
    "Real-Time Data Synchronization",
]

score = score(resume, kw)
