from flask import Flask, render_template, request, jsonify
import pickle
import re
import math
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
with open("models/logistic_regression.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/sgd_logistic_regression.pkl", "rb") as f:
    model_sgd = pickle.load(f)

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

# --- Halaman Utama Gabungan (Navbar, Tabel, Form Prediksi, Hasil) ---
@app.route("/", methods=["GET"])
def index():
    selected_perawi = request.args.get("perawi", PERAWI_TABLES[0])  # Default: perawi pertama
    if selected_perawi not in PERAWI_TABLES:
        selected_perawi = PERAWI_TABLES[0]
    
    search_query = request.args.get("search", "").strip()
    page = request.args.get("page", 1, type=int)
    per_page = 5
    offset = (page - 1) * per_page

    # Build where clause and params
    where_clause = ""
    params = []

    if search_query:
        if selected_perawi in ["hadits_training", "hadits_testing"]:
            where_clause = "WHERE hadis LIKE %s"
            params.append(f"%{search_query}%")
        else:
            where_clause = "WHERE nomor LIKE %s OR arab LIKE %s OR id LIKE %s"
            params.extend([f"%{search_query}%"] * 3)

    # Count total data
    count_query = f"SELECT COUNT(*) as total FROM {selected_perawi} {where_clause}"
    cursor.execute(count_query, params)
    total_hadits = cursor.fetchone()["total"]
    total_pages = math.ceil(total_hadits / per_page)

    # Get paginated data
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
    
    # Convert string "1"/"0" to integers for the template
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
                         search_query=search_query)



@app.route("/predik", methods=["POST"])
def prediksi():
    data = request.get_json()
    input_text = data.get("text", "").strip()

    # Validasi input kosong
    if not input_text:
        return jsonify({"error": "Teks hadis tidak boleh kosong!"}), 400

    # Preprocessing teks
    processed_text = full_preprocess(input_text)

    # Transformasi ke bentuk TF-IDF
    text_tfidf = tfidf_vectorizer.transform([processed_text])

    # Prediksi label biner
    prediction = model_sgd.predict(text_tfidf).tolist()[0]

    # Prediksi probabilitas untuk masing-masing label
    probabilities = model_sgd.predict_proba(text_tfidf)

    result = {
        "anjuran": bool(prediction[0]),
        "larangan": bool(prediction[1]),
        "informasi": bool(prediction[2]),
        "probabilitas": {
            "anjuran": float(probabilities[0][0][1]),
            "larangan": float(probabilities[1][0][1]),
            "informasi": float(probabilities[2][0][1])
        },
        "processed_text": processed_text
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
