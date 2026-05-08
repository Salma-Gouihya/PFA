import streamlit as st
import joblib, re, nltk, speech_recognition as sr, io, os, json
from datetime import datetime
import pandas as pd
import numpy as np

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

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    /* Global Styles */
    #MainMenu, footer {visibility: hidden;}
    .stApp {
        background: radial-gradient(circle at top right, #1e1e2e, #111119)!important;
        color: #e5e5e5;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {width: 8px;}
    ::-webkit-scrollbar-track {background: #111;}
    ::-webkit-scrollbar-thumb {background: #E50914; border-radius: 10px;}

    /* Glassmorphism Components */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        transition: all 0.3s ease;
    }

    /* Hero Section Upgraded */
    .hero {
        position: relative;
        height: 550px;
        border-radius: 20px;
        overflow: hidden;
        margin: 20px 48px 40px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.5);
    }
    .hero img {width: 100%; height: 100%; object-fit: cover; filter: brightness(0.5);}
    .hero-grad {
        position: absolute;
        inset: 0;
        background: linear-gradient(to right, rgba(17,17,25,0.95) 20%, rgba(17,17,25,0.4) 50%, transparent);
    }
    .hero-content {
        position: absolute;
        bottom: 80px;
        left: 60px;
        max-width: 650px;
        animation: fadeInUp 1s ease-out;
    }
    @keyframes fadeInUp {
        from {opacity: 0; transform: translateY(30px);}
        to {opacity: 1; transform: translateY(0);}
    }
    .hero-badge {
        background: linear-gradient(90deg, #E50914, #ff4d4d);
        color: #fff;
        padding: 5px 15px;
        border-radius: 50px;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 20px;
    }

    /* Film Cards */
    .film-card {
        background: rgba(255,255,255,0.02);
        border-radius: 16px;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 1px solid rgba(255,255,255,0.05);
        height: 100%;
        position: relative;
    }
    .film-card:hover {
        transform: scale(1.03);
        background: rgba(255,255,255,0.05);
        border-color: rgba(229,9,20,0.3);
        box-shadow: 0 15px 35px rgba(0,0,0,0.4), 0 0 20px rgba(229,9,20,0.1);
    }
    .film-card img {
        width: 100%;
        height: 300px;
        object-fit: cover;
        transition: transform 0.5s ease;
    }
    .film-card:hover img {transform: scale(1.1);}
    
    .card-body {padding: 18px;}
    .card-title {font-size: 18px; font-weight: 700; color: #fff; margin-bottom: 8px;}
    .card-genre {color: #E50914; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 1.5px;}

    /* Comment Cards Upgraded */
    .comment-card {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        border-left: 4px solid #444;
        backdrop-filter: blur(5px);
    }
    .comment-card.joy { border-left-color: #00ff88; box-shadow: inset 50px 0 50px -50px rgba(0,255,136,0.1); }
    .comment-card.anger { border-left-color: #ff4d4d; box-shadow: inset 50px 0 50px -50px rgba(255,77,77,0.1); }
    .comment-card.sadness { border-left-color: #00d4ff; box-shadow: inset 50px 0 50px -50px rgba(0,212,255,0.1); }
    .comment-card.fear { border-left-color: #bd00ff; box-shadow: inset 50px 0 50px -50px rgba(189,0,255,0.1); }
    .comment-card.surprise { border-left-color: #ffaa00; box-shadow: inset 50px 0 50px -50px rgba(255,170,0,0.1); }
    .comment-card.neutral { border-left-color: #888; }

    /* Admin Dashboard Metrics */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03)!important;
        border: 1px solid rgba(255,255,255,0.05)!important;
        border-radius: 15px!important;
        padding: 20px!important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    /* Buttons and Inputs */
    .stButton>button {
        background: linear-gradient(90deg, #E50914, #b20710)!important;
        color: white!important;
        border: none!important;
        padding: 12px 24px!important;
        border-radius: 8px!important;
        font-weight: 700!important;
        letter-spacing: 0.5px!important;
        box-shadow: 0 4px 15px rgba(229,9,20,0.3)!important;
    }
    .stTextArea textarea {
        background: rgba(255,255,255,0.05)!important;
        border: 1px solid rgba(255,255,255,0.1)!important;
        border-radius: 12px!important;
        color: white!important;
    }

    /* Titles */
    .section-title {
        font-size: 24px;
        font-weight: 800;
        color: #fff;
        padding: 0 48px;
        margin: 30px 0 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .section-title i { color: #E50914; }
</style>
""", unsafe_allow_html=True)

# ── Models ──────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        if os.path.exists('sentiment_model.pkl') and os.path.exists('vectorizer.pkl'):
            return joblib.load('sentiment_model.pkl'), joblib.load('vectorizer.pkl')
    except: pass
    return None, None

model, vectorizer = load_models()

# ── Data ─────────────────────────────────────────────────────────────
COMMENTS_FILE = "comments.json"

def load_comments():
    if os.path.exists(COMMENTS_FILE):
        try:
            with open(COMMENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except: pass
    return []

def save_comment(film_id, film_title, text, sentiment, style):
    comments = load_comments()
    comments.append({
        "id": len(comments) + 1,
        "film_id": film_id,
        "film_title": film_title,
        "text": text,
        "sentiment": sentiment,
        "style": style,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    with open(COMMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(comments, f, ensure_ascii=False, indent=2)

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
        label, style, confidence = EMOTION_MAP.get(pred, ("❓ Inconnu", "neutral", 0))
        return label, style, confidence
    except:
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

def go_home():
    st.session_state.page = "home"
    st.session_state.film_id = None

def go_admin():
    st.session_state.page = "admin"

def go_detail(film_id):
    st.session_state.page = "detail"
    st.session_state.film_id = film_id


# ── Navigation (Sidebar - Always Visible) ──────────────────────────────────────────
with st.sidebar:
    st.markdown('<h1 style="color:#E50914;font-size:28px;font-weight:800;margin-bottom:20px;"><i class="fas fa-film"></i> CINESTREAM</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Navigation Buttons (Clean labels without emojis)
    nav_choice = st.radio("NAVIGATION", ["Catalogue Films", "Dashboard Admin"], label_visibility="collapsed")
    
    if "Catalogue Films" in nav_choice:
        if st.session_state.page == "admin":
            go_home(); st.rerun()
    else:
        if st.session_state.page != "admin":
            go_admin(); st.rerun()

    st.markdown("---")
    st.markdown('<div style="color:#888;font-size:14px;"><i class="fas fa-lightbulb" style="color:#f1c40f;"></i> <b>Astuce</b> : Utilisez le micro pour laisser des avis vocaux.</div>', unsafe_allow_html=True)

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
            save_comment(film["id"], film["title"], comment_input, f"{sentiment} ({conf}%)", style)
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
                save_comment(film["id"], film["title"], f"[Vocal] {transcribed}", f"{sentiment} ({conf}%)", style)
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
            if os.path.exists(COMMENTS_FILE): os.remove(COMMENTS_FILE)
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)