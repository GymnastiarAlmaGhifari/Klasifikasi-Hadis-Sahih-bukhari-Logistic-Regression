from flask import Flask, render_template, request, jsonify
import pickle
import re
import math
import json
import numpy as np
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import mysql.connector

app = Flask(__name__)

# --- Koneksi Database MySQL ---
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="mysql_hadits"
)
cursor = db.cursor(dictionary=True)

# --- Daftar Tabel Perawi ---
PERAWI_TABLES = [
    "hadits_training",
    "hadits_testing",
    "hadits_bukhari",
    "hadits_muslim",
    "hadits_ahmad",
    "hadits_abu_daud",
    "hadits_tirmidzi",
    "hadits_ibnu_majah",
    "hadits_nasai",
    "hadits_malik",
    "hadits_darimi"
]

# --- Load Model & TF-IDF ---
with open("models/sgd_logistic_regression.pkl", "rb") as f:
    model_sgd = pickle.load(f)

with open("models/LR_manual_jurnal.pkl", "rb") as f:
    manual_model = pickle.load(f)  # {'anjuran': (W, b), 'larangan': ..., ...}

# Fungsi sigmoid dan dot product untuk prediksi manual
def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def dot(A, B):
    return sum(a * b for a, b in zip(A, B))

with open("data/tfidf/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# --- Preprocessing ---
def case_folding(text):
    return text.lower()

def remove_noise(text):
    return re.sub(r"[^a-z\s]", "", text)

def tokenize(text):
    return text.split()

stopword_factory = StopWordRemoverFactory()
stopwords = set(stopword_factory.get_stop_words())

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stopwords]

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

def stem_tokens(tokens):
    return [stemmer.stem(word) for word in tokens]

def full_preprocess(text):
    text = case_folding(text)
    text = remove_noise(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    return " ".join(tokens)

# --- Halaman Utama ---
@app.route("/", methods=["GET"])
def index():
    # Baca file hasil evaluasi
    try:
        with open('notebooks/result_evaluation_manual.json', 'r') as f:
            evaluation_results = json.load(f)
    except FileNotFoundError:
        evaluation_results = None
    except json.JSONDecodeError:
        evaluation_results = None

    selected_perawi = request.args.get("perawi", PERAWI_TABLES[0])
    if selected_perawi not in PERAWI_TABLES:
        selected_perawi = PERAWI_TABLES[0]
    
    search_query = request.args.get("search", "").strip()
    page = request.args.get("page", 1, type=int)
    per_page = 5
    offset = (page - 1) * per_page

    where_clause = ""
    params = []

    if search_query:
        if selected_perawi in ["hadits_training", "hadits_testing"]:
            where_clause = "WHERE hadis LIKE %s"
            params.append(f"%{search_query}%")
        else:
            where_clause = "WHERE nomor LIKE %s OR arab LIKE %s OR id LIKE %s"
            params.extend([f"%{search_query}%"] * 3)

    count_query = f"SELECT COUNT(*) as total FROM {selected_perawi} {where_clause}"
    cursor.execute(count_query, params)
    total_hadits = cursor.fetchone()["total"]
    total_pages = math.ceil(total_hadits / per_page)

    if selected_perawi in ["hadits_training", "hadits_testing"]:
        query = f"""
            SELECT hadis, anjuran, larangan, informasi
            FROM {selected_perawi}
            {where_clause}
            LIMIT %s OFFSET %s
        """
    else:
        query = f"""
            SELECT nomor, arab, id
            FROM {selected_perawi}
            {where_clause}
            LIMIT %s OFFSET %s
        """

    cursor.execute(query, params + [per_page, offset])
    hadits = cursor.fetchall()
    
    if selected_perawi in ["hadits_training", "hadits_testing"]:
        for hadith in hadits:
            hadith["anjuran"] = int(hadith["anjuran"])
            hadith["larangan"] = int(hadith["larangan"])
            hadith["informasi"] = int(hadith["informasi"])

    return render_template("index.html",
                         perawi_list=PERAWI_TABLES,
                         selected_perawi=selected_perawi,
                         hadits=hadits,
                         page=page,
                         total_pages=total_pages,
                         total_hadits=total_hadits,
                         search_query=search_query,
                         evaluation_results=evaluation_results)  # Tambahkan ini

# --- Endpoint Prediksi ---
# @app.route("/predik", methods=["POST"])
# def prediksi():
#     try:
#         data = request.get_json()
#         input_text = data.get("text", "").strip()

#         if not input_text:
#             return jsonify({"error": "Teks hadis tidak boleh kosong!"}), 400

#         processed_text = full_preprocess(input_text)
#         text_tfidf = tfidf_vectorizer.transform([processed_text])

#         # Get predictions
#         prediction = model_sgd.predict(text_tfidf).tolist()[0]
        
#         # Get probabilities for each class
#         probabilities = {
#             "anjuran": float(model_sgd.estimators_[0].predict_proba(text_tfidf)[0][1]),
#             "larangan": float(model_sgd.estimators_[1].predict_proba(text_tfidf)[0][1]),
#             "informasi": float(model_sgd.estimators_[2].predict_proba(text_tfidf)[0][1])
#         }

#         # Get decision function values
#         logits = {
#             "anjuran": float(model_sgd.estimators_[0].decision_function(text_tfidf)[0]),
#             "larangan": float(model_sgd.estimators_[1].decision_function(text_tfidf)[0]),
#             "informasi": float(model_sgd.estimators_[2].decision_function(text_tfidf)[0])
#         }

#         # Get top features
#         def get_top_features(class_idx, n=5):
#             coef = model_sgd.estimators_[class_idx].coef_[0]
#             top_indices = coef.argsort()[-n:][::-1]
#             feature_names = tfidf_vectorizer.get_feature_names_out()
#             return {feature_names[i]: float(coef[i]) for i in top_indices}

#         # Prepare TF-IDF features
#         tfidf_array = text_tfidf.toarray()[0]
#         feature_names = tfidf_vectorizer.get_feature_names_out()
#         tfidf_features = {feature_names[i]: float(tfidf_array[i]) 
#                          for i in tfidf_array.argsort()[-10:][::-1] if tfidf_array[i] > 0}

#         result = {
#             "success": True,
#             "prediction": {
#                 "anjuran": bool(prediction[0]),
#                 "larangan": bool(prediction[1]),
#                 "informasi": bool(prediction[2])
#             },
#             "probabilities": probabilities,
#             "logits": logits,
#             "processed_text": processed_text,
#             "tfidf_features": tfidf_features,
#             "top_coefficients": {
#                 "anjuran": get_top_features(0),
#                 "larangan": get_top_features(1),
#                 "informasi": get_top_features(2)
#             }
#         }

#         return jsonify(result)

#     except Exception as e:
#         return jsonify({
#             "success": False,
#             "error": f"Terjadi kesalahan: {str(e)}"
#         }), 500
@app.route("/predik", methods=["POST"])
def prediksi():
    try:
        data = request.get_json()
        input_text = data.get("text", "").strip()

        if not input_text:
            return jsonify({"error": "Teks hadis tidak boleh kosong!"}), 400

        processed_text = full_preprocess(input_text)
        text_tfidf = tfidf_vectorizer.transform([processed_text]).toarray()[0]

        prediction = {}
        probabilities = {}
        logits = {}

        for label in ["anjuran", "larangan", "informasi"]:
            W, b = manual_model[label]
            z = dot(text_tfidf, W) + b
            prob = sigmoid(z)
            pred = 1 if prob >= 0.5 else 0

            prediction[label] = bool(pred)
            probabilities[label] = float(prob)
            logits[label] = float(z)

        # TF-IDF Features
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_features = {
            feature_names[i]: float(text_tfidf[i])
            for i in np.argsort(text_tfidf)[-10:][::-1] if text_tfidf[i] > 0
        }

        result = {
            "success": True,
            "prediction": prediction,
            "probabilities": probabilities,
            "logits": logits,
            "processed_text": processed_text,
            "tfidf_features": tfidf_features,
            "top_coefficients": None  # Tidak tersedia karena tidak ada estimators_
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Terjadi kesalahan: {str(e)}"
        }), 500

# --- Endpoint Model Details ---
@app.route("/model-details", methods=["GET"])
def get_model_details():
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    def get_top_coefficients(class_idx, n=10):
        coef = model_sgd.coef_[class_idx]
        top_indices = coef.argsort()[-n:][::-1]
        return {feature_names[i]: float(coef[i]) for i in top_indices}
    
    return jsonify({
        "anjuran": get_top_coefficients(0),
        "larangan": get_top_coefficients(1),
        "informasi": get_top_coefficients(2),
        "feature_names": feature_names.tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)