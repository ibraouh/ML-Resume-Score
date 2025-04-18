from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")


@app.route("/")
def home():
    return "Resume Match API is running!"


@app.route("/match-score", methods=["POST"])
def match_score():
    data = request.json
    resume = data.get("resume")
    skills = data.get("skills")

    if not resume or not skills:
        return jsonify({"error": "Missing resume or skills"}), 400

    resume_embedding = model.encode(resume, convert_to_tensor=True)
    skills_embedding = model.encode(skills, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(resume_embedding, skills_embedding).item()
    match_score = round(similarity * 100)

    return jsonify({"match_score": match_score})
