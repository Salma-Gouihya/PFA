import streamlit as st
import joblib, re, nltk, speech_recognition as sr, io, os, json
from datetime import datetime
import pandas as pd
import numpy as np
import psycopg2
from passlib.hash import pbkdf2_sha256

try:
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    HAS_VISUALS = True
except ImportError:
    HAS_VISUALS = False


nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
# Garder les mots de négation pour une meilleure détection des émotions
stop_words -= {"not", "no", "nor", "never", "neither", "nobody", "nothing", "nowhere", "hardly", "scarcely", "barely"}

st.set_page_config(page_title="CineStream", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

# ── Theme Management ──────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

def toggle_theme():
    st.session_state.theme = "Light" if st.session_state.theme == "Dark" else "Dark"

is_dark = st.session_state.theme == "Dark"

# Base colors
bg_color = "radial-gradient(circle at top right, #1e1e2e, #111119)" if is_dark else "#f8f9fa"
text_color = "#e5e5e5" if is_dark else "#2d3436"
glass_bg = "rgba(255, 255, 255, 0.03)" if is_dark else "rgba(0, 0, 0, 0.02)"
glass_border = "rgba(255, 255, 255, 0.05)" if is_dark else "rgba(0, 0, 0, 0.05)"
card_bg = "rgba(255,255,255,0.02)" if is_dark else "#ffffff"
sidebar_bg = "#111119" if is_dark else "#ffffff"

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    /* Global Styles */
    #MainMenu, footer {{visibility: hidden;}}
    .stApp {{
        background: {bg_color}!important;
        color: {text_color}!important;
        font-family: 'Outfit', sans-serif;
    }}
    
    /* Force text color on all basic elements in Light mode */
    .stApp p, .stApp span, .stApp div, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
        color: {text_color};
    }}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {{width: 8px;}}
    ::-webkit-scrollbar-track {{background: {"#111" if is_dark else "#eee"};}}
    ::-webkit-scrollbar-thumb {{background: #E50914; border-radius: 10px;}}

    /* Glassmorphism Components */
    .glass-card {{
        background: {glass_bg};
        backdrop-filter: blur(10px);
        border: 1px solid {glass_border};
        border-radius: 15px;
        padding: 20px;
        transition: all 0.3s ease;
        color: {text_color};
    }}

    /* Hero Section */
    .hero {{
        position: relative;
        height: 550px;
        border-radius: 20px;
        overflow: hidden;
        margin: 20px 48px 40px;
        box-shadow: 0 20px 50px rgba(0,0,0,{"0.5" if is_dark else "0.1"});
    }}
    .hero img {{width: 100%; height: 100%; object-fit: cover; filter: brightness({"0.5" if is_dark else "0.8"});}}
    .hero-grad {{
        position: absolute;
        inset: 0;
        background: {"linear-gradient(to right, rgba(17,17,25,0.95) 20%, rgba(17,17,25,0.4) 50%, transparent)" if is_dark else "linear-gradient(to right, rgba(255,255,255,0.9) 20%, rgba(255,255,255,0.2) 50%, transparent)"};
    }}
    .hero-content {{
        position: absolute;
        bottom: 80px;
        left: 60px;
        max-width: 650px;
        animation: fadeInUp 1s ease-out;
    }}
    .hero-content h1, .hero-content p, .hero-content span {{ 
        color: {"#fff" if is_dark else "#111"}!important; 
    }}

    /* Film Cards */
    .film-card {{
        background: {card_bg};
        border-radius: 16px;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 1px solid {glass_border};
        height: 100%;
        position: relative;
        box-shadow: {"none" if is_dark else "0 10px 20px rgba(0,0,0,0.05)"};
    }}
    .film-card:hover {{
        transform: scale(1.03);
        background: {"rgba(255,255,255,0.05)" if is_dark else "#fff"};
        border-color: rgba(229,9,20,0.3);
        box-shadow: 0 15px 35px rgba(0,0,0,{"0.4" if is_dark else "0.1"}), 0 0 20px rgba(229,9,20,0.1);
    }}
    .card-title {{ color: {"#fff" if is_dark else "#111"}!important; }}
    .card-row {{ color: {"#ccc" if is_dark else "#555"}!important; }}
    .card-desc {{ color: {"#bbb" if is_dark else "#666"}!important; }}

    /* Comment Cards */
    .comment-card {{
        background: {glass_bg};
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        border-left: 4px solid #444;
        backdrop-filter: blur(5px);
        color: {text_color};
    }}
    .comment-text {{ color: {text_color}!important; }}
    .comment-footer {{ color: {"#888" if is_dark else "#555"}!important; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg}!important;
    }}
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label {{
        color: {text_color}!important;
    }}

    /* Metrics */
    div[data-testid="stMetric"] {{
        background: {glass_bg}!important;
        border: 1px solid {glass_border}!important;
        border-radius: 15px!important;
        padding: 20px!important;
        color: {text_color}!important;
    }}
    div[data-testid="stMetricValue"] {{ color: {text_color}!important; }}
    div[data-testid="stMetricLabel"] {{ color: {"#888" if is_dark else "#444"}!important; }}
    
    .section-title {{ color: {"#fff" if is_dark else "#111"}!important; }}

    /* Buttons and Inputs */
    .stButton>button {{
        background: linear-gradient(90deg, #E50914, #b20710)!important;
        color: white!important;
        border: none!important;
        padding: 12px 24px!important;
        border-radius: 8px!important;
        font-weight: 700!important;
        letter-spacing: 0.5px!important;
        box-shadow: 0 4px 15px rgba(229,9,20,0.3)!important;
    }}
    /* Force white text inside buttons specifically */
    .stButton>button div, .stButton>button p, .stButton>button span {{
        color: white!important;
    }}

    .stTextArea textarea {{
        background: {"rgba(255,255,255,0.05)" if is_dark else "rgba(0,0,0,0.03)"}!important;
        border: 1px solid {"rgba(255,255,255,0.1)" if is_dark else "rgba(0,0,0,0.1)"}!important;
        border-radius: 12px!important;
        color: {text_color}!important;
    }}
</style>
""", unsafe_allow_html=True)

# ── Models ──────────────────────────────────────────────────────────
# @st.cache_resource
def load_models():
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, 'sentiment_model.pkl')
        vec_path = os.path.join(base_path, 'vectorizer.pkl')
        if os.path.exists(model_path) and os.path.exists(vec_path):
            return joblib.load(model_path), joblib.load(vec_path)
    except Exception as e:
        print(f"ERROR LOADING MODELS: {e}")
    return None, None

model, vectorizer = load_models()
if model is None:
    st.error("⚠️ ERREUR : Le modèle d'IA n'est pas chargé. Vérifiez les logs.")
else:
    st.success("✅ Modèle d'IA chargé avec succès !")

# ── Data ─────────────────────────────────────────────────────────────
# ── Database ────────────────────────────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "user": "postgres",
    "password": "Goui3006",
    "dbname": "Goui3006"
}

def get_db_connection():
    try:
        return psycopg2.connect(**DB_CONFIG)
    except:
        return None

def load_comments(film_id=None):
    conn = get_db_connection()
    if not conn: return []
    try:
        cur = conn.cursor()
        if film_id:
            cur.execute("SELECT id, film_id, film_title, text, sentiment, style, date FROM comments WHERE film_id = %s ORDER BY date DESC", (film_id,))
        else:
            cur.execute("SELECT id, film_id, film_title, text, sentiment, style, date FROM comments ORDER BY date DESC")
        rows = cur.fetchall()
        comments = []
        for r in rows:
            comments.append({
                "id": r[0], "film_id": r[1], "film_title": r[2],
                "text": r[3], "sentiment": r[4], "style": r[5],
                "date": r[6].strftime("%Y-%m-%d %H:%M") if hasattr(r[6], 'strftime') else r[6]
            })
        cur.close()
        conn.close()
        return comments
    except:
        return []

def save_comment(film_id, film_title, text, sentiment, style, user_id=None):
    conn = get_db_connection()
    if not conn: return False
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO comments (film_id, film_title, text, sentiment, style, user_id)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (film_id, film_title, text, sentiment, style, user_id))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except:
        return False

# ── Authentication ────────────────────────────────────────────────────
def authenticate_user(username, password):
    conn = get_db_connection()
    if not conn: return None
    cur = conn.cursor()
    cur.execute("SELECT id, username, password, role FROM users WHERE username = %s", (username,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    if user and pbkdf2_sha256.verify(password, user[2]):
        return {"id": user[0], "username": user[1], "role": user[3]}
    return None

def register_user(username, password):
    conn = get_db_connection()
    if not conn: return False
    try:
        cur = conn.cursor()
        hashed_pw = pbkdf2_sha256.hash(password)
        cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_pw))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except:
        return False

FILMS = [
    {"id":1,"title":"Interstellar","year":2014,"director":"Christopher Nolan","genre":"Sci-Fi","duration":"2h 49min","rating":8.7,"votes":"2.1M","description":"Une équipe d'astronautes voyage à travers un trou de ver pour trouver une nouvelle demeure pour l'humanité.","synopsis":"Alors que la Terre se meurt, un groupe d'explorateurs utilise un tunnel spatio-temporel pour dépasser les limites des voyages dans l'espace et parcourir les distances astronomiques qui séparent la Terre d'autres planètes.","cast":"Matthew McConaughey, Anne Hathaway, Jessica Chastain","poster":"https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg","backdrop":"https://image.tmdb.org/t/p/w1280/xJHokMbljvjADYdit5fK5VQsXEG.jpg"},
    {"id":2,"title":"Get Out","year":2017,"director":"Jordan Peele","genre":"Horreur","duration":"1h 44min","rating":7.7,"votes":"684K","description":"Un homme découvre les terrifiants secrets de la famille de sa petite amie lors d'un week-end en famille.","synopsis":"Chris, un jeune Afro-Américain, rend visite pour la première fois à la famille de sa petite amie Rose. Ce qui semble être un week-end ordinaire vire rapidement au cauchemar.","cast":"Daniel Kaluuya, Allison Williams, Bradley Whitford","poster":"https://image.tmdb.org/t/p/w500/tFXcEccSQMf3lfhfXKSU9iRBpa3.jpg","backdrop":"https://image.tmdb.org/t/p/w1280/n3ONrLRuZbM5JGvjHZA4NOflQpG.jpg"},
    {"id":3,"title":"Inception","year":2010,"director":"Christopher Nolan","genre":"Thriller","duration":"2h 28min","rating":8.8,"votes":"2.5M","description":"Un voleur pénètre dans les rêves de ses cibles pour en extraire des secrets, et reçoit une mission impossible.","synopsis":"Dom Cobb est capable de s'introduire dans les rêves pour voler des secrets. On lui propose une mission encore plus dangereuse : planter une idée dans l'esprit d'une cible — l'inception.","cast":"Leonardo DiCaprio, Joseph Gordon-Levitt, Elliot Page","poster":"https://image.tmdb.org/t/p/w500/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg","backdrop":"https://image.tmdb.org/t/p/w1280/s2bT29y0ngXxxu2IA8AOzzXTRhd.jpg"},
    {"id":4,"title":"Parasite","year":2019,"director":"Bong Joon-ho","genre":"Drame","duration":"2h 12min","rating":8.6,"votes":"906K","description":"Une famille pauvre infiltre progressivement la vie luxueuse d'une famille aisée en Corée du Sud.","synopsis":"La famille Kim, désargentée, s'infiltre habilement dans la vie d'une riche famille. Cette cohabitation va déclencher une série d'événements inattendus et violents.","cast":"Song Kang-ho, Lee Sun-kyun, Cho Yeo-jeong","poster":"https://image.tmdb.org/t/p/w500/7IiTTgloJzvGI1TAYymCfbfl3vT.jpg","backdrop":"https://image.tmdb.org/t/p/w1280/ApiBzeaa95TNYLisFOjsRq0gYmF.jpg"},
    {"id":5,"title":"The Dark Knight","year":2008,"director":"Christopher Nolan","genre":"Action","duration":"2h 32min","rating":9.0,"votes":"2.8M","description":"Batman affronte le Joker, un génie du crime qui plonge Gotham City dans le chaos total.","synopsis":"Batman, le Commissaire Gordon et Harvey Dent s'allient contre le crime organisé. Mais l'émergence du Joker, un criminel imprévisible, va tout bouleverser.","cast":"Christian Bale, Heath Ledger, Aaron Eckhart","poster":"https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg","backdrop":"https://image.tmdb.org/t/p/w1280/hqkIcbrOHL86UncnHIsHVcVmzue.jpg"},
    {"id":6,"title":"Hereditary","year":2018,"director":"Ari Aster","genre":"Horreur","duration":"2h 07min","rating":7.3,"votes":"380K","description":"Après la mort de sa grand-mère, une famille découvre des secrets terrifiants sur son héritage.","synopsis":"Quand Annie perd sa mère, sa famille commence à révéler des secrets de plus en plus obscurs sur leur lignée. Les événements qui suivent vont menacer leur santé mentale et leur vie.","cast":"Toni Collette, Milly Shapiro, Gabriel Byrne","poster":"https://image.tmdb.org/t/p/w500/4HWAQu28e2yaWrtupFPGFkdNU7V.jpg","backdrop":"https://image.tmdb.org/t/p/w1280/5vwPi6yPGGCqWCJAVi64r5FXQE.jpg"},
]

# ── NLP ──────────────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zàâäéèêëîïôùûüç\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

# Mapping émotion → (label affiché, classe CSS)
EMOTION_MAP = {
    "joy":      ("😊 Content",     "joy"),
    "anger":    ("😠 En colère",   "anger"),
    "sadness":  ("😢 Triste",      "sadness"),
    "fear":     ("😨 Peur",        "fear"),
    "surprise": ("😲 Surpris",     "surprise"),
    "neutral":  ("😐 Neutre",       "neutral"),
}

def predict(text):
    if not text or not text.strip() or model is None:
        return "❓ Inconnu", "neutral", 0
    try:
        clean = clean_text(text)
        vec = vectorizer.transform([clean])
        
        # Obtenir les scores de décision pour calculer la confiance
        decision = model.decision_function(vec)[0]
        # Normalisation Softmax simplifiée pour avoir un score entre 0 et 100
        exp_scores = np.exp(decision - np.max(decision))
        probs = exp_scores / exp_scores.sum()
        confidence = int(np.max(probs) * 100)
        
        pred = model.predict(vec)[0]
        label, style = EMOTION_MAP.get(pred, ("❓ Inconnu", "neutral"))
        return label, style, confidence
    except Exception as e:
        print(f"PREDICTION ERROR: {e}")
        return "❓ Inconnu", "neutral", 0

def get_top_keywords(emotion_name, top_n=10):
    if model is None or vectorizer is None:
        return []
    try:
        # Trouver l'index de l'émotion dans le modèle
        classes = model.classes_.tolist()
        if emotion_name not in classes:
            return []
        class_idx = classes.index(emotion_name)
        
        # Obtenir les noms des features (mots)
        feature_names = vectorizer.get_feature_names_out()
        # Obtenir les coefficients pour cette classe
        coefs = model.coef_[class_idx]
        
        # Trier par poids
        top_indices = np.argsort(coefs)[::-1][:top_n]
        return [feature_names[i] for i in top_indices]
    except:
        return []

# ── Session State ─────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "film_id" not in st.session_state:
    st.session_state.film_id = None
if "user" not in st.session_state:
    st.session_state.user = None

def go_home():
    st.session_state.page = "home"
    st.session_state.film_id = None

def go_admin():
    if st.session_state.user and st.session_state.user['role'] == 'admin':
        st.session_state.page = "admin"
    else:
        st.error("Accès réservé aux administrateurs.")

def go_detail(film_id):
    st.session_state.page = "detail"
    st.session_state.film_id = film_id


# ── Navigation (Sidebar - Always Visible) ──────────────────────────────────────────
with st.sidebar:
    st.markdown('<h1 style="color:#E50914;font-size:28px;font-weight:800;margin-bottom:20px;"><i class="fas fa-film"></i> CINESTREAM</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sélecteur de Thème
    st.markdown("---")
    theme_icon = "☀️" if is_dark else "🌙"
    theme_label = "Mode Clair" if is_dark else "Mode Sombre"
    if st.button(f"{theme_icon} {theme_label}", use_container_width=True):
        toggle_theme()
        st.rerun()

    # Section Authentification
    st.markdown("---")
    if st.session_state.user:
        st.markdown(f"👤 Connecté : **{st.session_state.user['username']}**")
        st.markdown(f"🏷️ Rôle : `{st.session_state.user['role'].upper()}`")
        if st.button("🚪 Se déconnecter", use_container_width=True):
            st.session_state.user = None
            st.session_state.page = "home"
            st.rerun()
    else:
        with st.expander("🔐 Connexion / Inscription", expanded=True):
            auth_mode = st.radio("Action", ["Connexion", "Inscription"], label_visibility="collapsed")
            u_input = st.text_input("Utilisateur", key="auth_u")
            p_input = st.text_input("Mot de passe", type="password", key="auth_p")
            
            if auth_mode == "Connexion":
                if st.button("Se connecter", use_container_width=True):
                    user = authenticate_user(u_input, p_input)
                    if user:
                        st.session_state.user = user
                        st.success(f"Bienvenue {u_input} !")
                        st.rerun()
                    else:
                        st.error("Identifiants incorrects.")
            else:
                if st.button("Créer un compte", use_container_width=True):
                    if u_input and p_input:
                        if register_user(u_input, p_input):
                            st.success("Compte créé ! Connectez-vous.")
                        else:
                            st.error("Pseudo déjà pris.")
                    else:
                        st.warning("Remplissez les champs.")

    st.markdown("---")
    
    # Navigation
    if st.button("🏠 Accueil / Catalogue", use_container_width=True):
        go_home()
        st.rerun()
        
    if st.session_state.user and st.session_state.user['role'] == 'admin':
        if st.button("📊 Dashboard Admin", use_container_width=True):
            go_admin()
            st.rerun()

    st.markdown("---")
    st.markdown('<div style="color:#888;font-size:13px;text-align:center;">Powered by PostgreSQL</div>', unsafe_allow_html=True)

# Logo en haut de page pour le look
st.markdown('<div style="padding:10px 0;text-align:center;"><h1 style="color:#E50914;font-size:32px;font-weight:800;"><i class="fas fa-play-circle"></i> CINESTREAM</h1></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE : HOME
# ══════════════════════════════════════════════════════════════════════
if st.session_state.page == "home":
    featured = FILMS[4]  # The Dark Knight as hero
    st.markdown(f"""
    <div class="hero">
        <img src="{featured['backdrop']}" />
        <div class="hero-grad"></div>
        <div class="hero-content">
            <span class="hero-badge"><i class="fas fa-star"></i> {featured['rating']} · RECOMMANDÉ</span>
            <h1 style="font-size:56px; font-weight:800; color:white; margin:0 0 15px;">{featured['title']}</h1>
            <div style="font-size:14px; color:#ccc; margin-bottom:15px; display:flex; gap:20px;">
                <span><i class="far fa-calendar"></i> {featured['year']}</span>
                <span><i class="far fa-clock"></i> {featured['duration']}</span>
                <span><i class="fas fa-tag"></i> {featured['genre']}</span>
            </div>
            <p style="font-size:16px; color:#bbb; line-height:1.6; margin-bottom:25px;">{featured['description']}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_hero, _ = st.columns([2, 5])
    with col_hero:
        if st.button(f"Regarder {featured['title']}", key="hero_btn"):
            go_detail(featured['id']); st.rerun()

    st.markdown('<div class="section-title"><i class="fas fa-fire"></i> Films Populaires</div>', unsafe_allow_html=True)

    with st.container():
        cols = st.columns(3, gap="medium")
        for i, film in enumerate(FILMS):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="film-card">
                    <img src="{film['poster']}" alt="{film['title']}" />
                    <div class="card-body">
                        <div class="card-genre">{film['genre']}</div>
                        <div class="card-title">{film['title']}</div>
                        <div class="card-row">
                            <span>{film['year']} · {film['duration']}</span>
                            <span class="card-rating"><i class="fas fa-star"></i> {film['rating']}</span>
                        </div>
                        <div class="card-desc">{film['description']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Voir Détails", key=f"card_{film['id']}"):
                    st.session_state.film_id = film['id']
                    st.session_state.page = "detail"
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════
# PAGE : DETAIL
# ══════════════════════════════════════════════════════════════════════
elif st.session_state.page == "detail":
    film = next((f for f in FILMS if f["id"] == st.session_state.film_id), FILMS[0])
    all_comments = [c for c in load_comments() if c["film_id"] == film["id"]]

    st.markdown(f"""
    <div class="detail-backdrop">
        <img src="{film['backdrop']}" />
        <div class="detail-grad"></div>
        <div class="detail-overlay">
            <img class="detail-poster" src="{film['poster']}" />
            <div>
                <span class="badge"><i class="fas fa-tag"></i> {film['genre']}</span>
                <h1 class="detail-title">{film['title']}</h1>
                <div class="detail-meta">
                    <span><i class="far fa-calendar-alt"></i> {film['year']}</span>
                    <span><i class="far fa-clock"></i> {film['duration']}</span>
                    <span><i class="fas fa-star" style="color:#f5c518;"></i> {film['rating']}/10</span>
                    <span><i class="fas fa-user-tie"></i> {film['director']}</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    back_col, _ = st.columns([1, 6])
    with back_col:
        if st.button("↩ Retour au catalogue", key="back"):
            go_home(); st.rerun()

    st.markdown(f"""
    <div class="detail-body">
        <div class="synopsis-title"><i class="fas fa-align-left"></i> Synopsis</div>
        <p class="synopsis-text">{film['synopsis']}</p>
        <div class="info-grid">
            <div class="info-item"><label><i class="fas fa-video"></i> Réalisateur</label><span>{film['director']}</span></div>
            <div class="info-item"><label><i class="fas fa-users"></i> Casting</label><span>{film['cast']}</span></div>
            <div class="info-item"><label><i class="fas fa-film"></i> Genre</label><span>{film['genre']}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding:0 60px'>", unsafe_allow_html=True)

    st.markdown('<div class="comment-section-title"><i class="fas fa-edit"></i> Laisser un avis</div>', unsafe_allow_html=True)
    comment_input = st.text_area("Votre commentaire :", placeholder="Partagez votre avis...", height=100, key="txt_comment")
    if st.button("Envoyer l'avis", key="send_comment"):
        if comment_input.strip():
            sentiment, style, conf = predict(comment_input)
            user_id = st.session_state.user['id'] if st.session_state.user else None
            save_comment(film["id"], film["title"], comment_input, f"{sentiment} ({conf}%)", style, user_id)
            st.success(f"Avis enregistré !")
            st.rerun()

    st.markdown('<div class="comment-section-title"><i class="fas fa-microphone"></i> Avis vocal</div>', unsafe_allow_html=True)
    audio_data = st.audio_input("Enregistrer", key=f"audio_{film['id']}")
    if audio_data is not None:
        st.audio(audio_data)
        with st.spinner("Analyse en cours..."):
            try:
                recognizer = sr.Recognizer()
                with sr.AudioFile(audio_data) as source:
                    recorded = recognizer.record(source)
                transcribed = recognizer.recognize_google(recorded, language="fr-FR")
                st.info(f"Transcription : {transcribed}")
                sentiment, style, conf = predict(transcribed)
                user_id = st.session_state.user['id'] if st.session_state.user else None
                save_comment(film["id"], film["title"], f"[Vocal] {transcribed}", f"{sentiment} ({conf}%)", style, user_id)
                st.success(f"Avis vocal enregistré !")
                st.rerun()
            except:
                st.error("Erreur de transcription.")

    st.markdown('<div class="comment-section-title"><i class="fas fa-comments"></i> Avis récents</div>', unsafe_allow_html=True)
    if not all_comments:
        st.markdown('<div class="no-comments">Pas encore d\'avis.</div>', unsafe_allow_html=True)
    else:
        for c in reversed(all_comments):
            style_cls = c.get("style", "neutral")
            st.markdown(f"""
            <div class="comment-card {style_cls}">
                <div class="comment-text">{c['text']}</div>
                <div class="comment-footer">
                    <span>{c['date']}</span>
                    <span class="s-badge {style_cls}">{c['sentiment']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE : ADMIN
# ══════════════════════════════════════════════════════════════════════
elif st.session_state.page == "admin":
    import pandas as pd
    st.markdown("<div style='padding:32px 60px'>", unsafe_allow_html=True)
    st.markdown('## <i class="fas fa-chart-line"></i> Dashboard Administrateur', unsafe_allow_html=True)

    comments = load_comments()
    if not comments:
        st.info("Aucun retour pour le moment.")
    else:
        df = pd.DataFrame(comments)

        # ── Métriques globales ─────────────────────────────────────────
        EMOTION_COLORS = {
            "joy":      "#2ecc71",
            "anger":    "#e74c3c",
            "sadness":  "#3498db",
            "fear":     "#9b59b6",
            "surprise": "#f39c12",
            "neutral":  "#888888",
        }
        EMOTION_ICONS = {
            "joy":      "😊", "anger": "😠", "sadness": "😢",
            "fear":     "😨", "surprise": "😲", "neutral": "❓",
        }

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        cols_metrics = [m1, m2, m3, m4, m5, m6]
        emotions_order = ["joy", "anger", "sadness", "fear", "surprise", "neutral"]
        labels_fr = {
            "joy": "Content", "anger": "En colère", "sadness": "Triste",
            "fear": "Peur", "surprise": "Surpris", "neutral": "Neutre"
        }
        for i, emo in enumerate(emotions_order):
            count = len(df[df["style"] == emo])
            cols_metrics[i].metric(f"{EMOTION_ICONS[emo]} {labels_fr[emo]}", count)

        st.markdown("---")

        # ── Distribution des émotions (camembert) ──────────────────────
        emo_counts = df["style"].value_counts()
        if len(emo_counts) > 0:
            st.markdown('<div class="section-title"><i class="fas fa-chart-pie"></i> Visualisation des sentiments</div>', unsafe_allow_html=True)
            col_pie, col_bar = st.columns([1, 1.5], gap="large")
            with col_pie:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<h4>Répartition Globale</h4>', unsafe_allow_html=True)
                if HAS_VISUALS:
                    fig, ax = plt.subplots(figsize=(4, 4), facecolor='none')
                    pie_labels = [f"{EMOTION_ICONS.get(e,'❓')} {labels_fr.get(e, e)}" for e in emo_counts.index]
                    pie_colors = [EMOTION_COLORS.get(e, '#888') for e in emo_counts.index]
                    ax.pie(emo_counts.values, labels=pie_labels, colors=pie_colors,
                           autopct='%1.0f%%', textprops={'color': '#e5e5e5', 'fontsize': 10},
                           wedgeprops={'linewidth': 2, 'edgecolor': '#111'})
                    st.pyplot(fig)
                    plt.close()
                else:
                    # Alternative simple si matplotlib manque
                    st.bar_chart(emo_counts)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_bar:
                st.markdown('<div class="glass-card" style="height:100%">', unsafe_allow_html=True)
                st.markdown('<h4>Émotions par Film</h4>', unsafe_allow_html=True)
                chart_data = df.groupby(['film_title', 'style']).size().unstack(fill_value=0)
                ordered_cols = [e for e in emotions_order if e in chart_data.columns]
                chart_data = chart_data[ordered_cols]
                st.bar_chart(chart_data, color=[EMOTION_COLORS.get(c, '#888') for c in chart_data.columns])
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Top films par émotion ──────────────────────────────────────
        st.markdown('### <i class="fas fa-trophy"></i> Analyse par émotion dominante', unsafe_allow_html=True)
        emo_cols = st.columns(3)
        shown = [("joy", "#2ecc71", "fas fa-smile"), ("anger", "#e74c3c", "fas fa-angry"), ("sadness", "#3498db", "fas fa-sad-tear")]
        for idx, (emo, color, icon) in enumerate(shown):
            with emo_cols[idx]:
                emo_df = df[df["style"] == emo]
                st.markdown(f'<div style="color:{color};font-weight:700;font-size:14px;"><i class="{icon}"></i> {labels_fr[emo]} ({len(emo_df)})</div>', unsafe_allow_html=True)
                top = emo_df.groupby("film_title").size().sort_values(ascending=False).head(3)
                for title, cnt in top.items():
                    st.markdown(f'<div style="color:#ccc;font-size:13px;padding:2px 0;">• <b>{title}</b> — {cnt}</div>', unsafe_allow_html=True)
                
                # Mots clés associés
                keywords = get_top_keywords(emo, 5)
                if keywords:
                    st.markdown(f'<div style="color:#666;font-size:11px;font-style:italic;margin-top:5px;">Mots clés : {", ".join(keywords)}</div>', unsafe_allow_html=True)
                
                if len(top) == 0:
                    st.markdown('<div style="color:#555;font-size:13px;">Aucun avis</div>', unsafe_allow_html=True)

        st.markdown("---")
        
        # ── Word Cloud ────────────────────────────────────────────────
        st.markdown('### <i class="fas fa-cloud"></i> Nuage de mots global', unsafe_allow_html=True)
        all_text = " ".join(df["text"].values)
        if len(all_text) > 10 and HAS_VISUALS:
            try:
                wc = WordCloud(width=800, height=400, background_color='#141414', 
                               colormap='Reds', font_path=None).generate(all_text)
                fig_wc, ax_wc = plt.subplots(figsize=(10, 5), facecolor='#141414')
                ax_wc.imshow(wc, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc)
                plt.close()
            except:
                st.info("Erreur lors de la génération du nuage de mots.")
        else:
            st.info("Visualisations avancées indisponibles ou pas assez de données.")

        st.markdown("---")

        # ── Historique complet ─────────────────────────────────────────
        st.markdown('### <i class="fas fa-table"></i> Historique des avis', unsafe_allow_html=True)
        df_display = df[["date", "film_title", "text", "sentiment"]].copy()
        df_display.columns = ["Date", "Film", "Commentaire", "Émotion détectée"]
        st.dataframe(df_display.sort_values("Date", ascending=False), use_container_width=True)

        if st.button("Réinitialiser les données"):
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("DELETE FROM comments")
            conn.commit()
            cur.close()
            conn.close()
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)