"""
Script d'entraînement : Modèle de détection d'émotions multi-classes (v2)
==========================================================================
Version améliorée avec un dataset beaucoup plus large et varié,
incluant des phrases courtes pour une meilleure généralisation.

5 émotions :
  - joy      (content / heureux)
  - anger    (en colère / frustré)
  - sadness  (triste / déçu)
  - fear     (peur / anxieux)
  - surprise (surpris / choqué)
"""

import re, os, joblib
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import nltk

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
# On retire "not", "no", "nor" des stopwords pour garder la négation
stop_words -= {"not", "no", "nor", "never", "neither", "nobody", "nothing", "nowhere", "hardly", "scarcely", "barely"}

# ══════════════════════════════════════════════════════════════════════════════
# DATASET : phrases variées par émotion (courtes + longues + FR + EN)
# ══════════════════════════════════════════════════════════════════════════════
SAMPLES = {
    "joy": [
        # --- Courtes (EN) ---
        "good movie", "great film", "so good", "amazing", "wonderful", "excellent",
        "I loved it", "really good", "the best", "fantastic", "brilliant",
        "very good", "pretty good", "nice movie", "awesome film", "superb",
        "I enjoyed it", "loved this", "perfect movie", "so fun", "delightful",
        "beautiful film", "outstanding", "incredible movie", "phenomenal",
        "magnificent", "splendid", "marvelous", "terrific", "impressive",
        "entertaining", "fun movie", "enjoyable", "satisfying", "great job",
        "well done", "masterpiece", "love it", "the best movie", "so happy",
        "made me smile", "feel good movie", "heartwarming",
        # --- Moyennes (EN) ---
        "This movie was absolutely wonderful and joyful",
        "I loved every moment of this film, it made me so happy",
        "Brilliant performance, I left the cinema with a huge smile",
        "This is a masterpiece, I feel great after watching it",
        "Amazing storyline, funny and uplifting, I enjoyed it so much",
        "The best movie I have seen in years, truly entertaining",
        "I laughed so hard, this film is pure joy",
        "Fantastic direction and a heartwarming story",
        "Excellent cast, the movie left me feeling euphoric",
        "I am delighted by this film, it exceeded all my expectations",
        "Such a fun and cheerful movie, I had a blast",
        "Great movie, beautiful visuals, I feel happy and inspired",
        "This film brought tears of joy to my eyes",
        "One of the most enjoyable movies I have ever watched",
        "Perfect movie night, I am so pleased with this choice",
        "Really good acting and a compelling story",
        "The movie was entertaining from start to finish",
        "A very good film with a positive message",
        "It was a pleasure to watch this movie",
        "Such a lovely and uplifting experience",
        "I would recommend this movie to everyone",
        "This movie made my day so much better",
        "What a great time I had watching this film",
        "The performances were absolutely top notch",
        "A truly wonderful cinematic experience",
        # --- Courtes (FR) ---
        "bon film", "tres bien", "excellent", "magnifique", "genial",
        "j'ai adore", "super film", "formidable", "parfait", "incroyable",
        "le meilleur", "fantastique", "merveilleux", "splendide", "bravo",
        "j'ai aime", "tres bon", "beau film", "superbe", "extraordinaire",
        # --- Moyennes (FR) ---
        "Ce film etait absolument magnifique et m'a rendu si heureux",
        "J'ai adore chaque instant c'etait une experience merveilleuse",
        "Excellente performance je suis reparti avec le sourire",
        "Un chef-d'oeuvre je me sens vraiment bien apres l'avoir vu",
        "Histoire incroyable drole et emouvante j'ai vraiment apprecie",
        "Le meilleur film que j'ai vu depuis des annees",
        "J'ai ri aux eclats quel bonheur de regarder ce film",
        "Mise en scene fantastique et une histoire touchante",
        "Excellent casting le film m'a laisse euphorique",
        "Je suis ravi par ce film il a depasse toutes mes attentes",
        "Film si amusant et joyeux je me suis vraiment amuse",
        "Super film visuellement beau je me sens heureux et inspire",
        "Ce film m'a apporte des larmes de joie",
        "Un des films les plus agreables que j'aie jamais vus",
        "Soiree cinema parfaite je suis tres satisfait de ce choix",
    ],

    "anger": [
        # --- Courtes (EN) ---
        "bad movie", "terrible", "awful", "horrible", "worst movie",
        "so bad", "really bad", "hate it", "I hated it", "disgusting",
        "trash", "garbage", "waste of time", "rubbish", "pathetic",
        "stupid movie", "dumb", "ridiculous", "absurd", "unacceptable",
        "not good", "very bad", "worst film", "pure trash", "total garbage",
        "annoying", "infuriating", "frustrating", "unbearable", "atrocious",
        "I despise this", "the worst", "complete disaster", "total failure",
        "made me angry", "so frustrated", "offensive", "insulting",
        "I regret watching this", "what a waste", "worthless",
        "not worth watching", "complete trash",
        # --- Courtes avec contexte movie/film (EN) ---
        "the movie is bad", "the movie is awful", "the movie is terrible",
        "the movie is horrible", "the film is bad", "the film is awful",
        "the film is terrible", "the film is horrible", "awful movie",
        "terrible movie", "horrible movie", "horrible film", "terrible film",
        "awful film", "this movie is bad", "this film is bad",
        "this movie is awful", "this film is awful", "this movie is terrible",
        "bad film", "worst movie ever", "worst film ever",
        "the movie is not good", "the film is not good",
        "movie was bad", "movie was awful", "movie was terrible",
        "film was bad", "film was awful", "film was terrible",
        "did not like this movie", "did not like this film",
        "I did not enjoy this movie", "not a good movie", "not a good film",
        "disappointing movie", "disappointing film", "a bad movie",
        "the movie sucks", "this movie sucks", "it was bad", "it was awful",
        "it was terrible", "it was horrible", "really awful", "truly awful",
        "absolutely terrible", "absolutely horrible", "absolutely awful",
        # --- Moyennes (EN) ---
        "This movie is a complete waste of time I am furious",
        "The director ruined everything I am so angry about this film",
        "Terrible plot offensive and deeply frustrating to watch",
        "I hate this movie what an insult to the audience",
        "This film made me incredibly angry poor storytelling",
        "I left the cinema outraged disgusting waste of money",
        "Boring and infuriating the worst film I have seen",
        "I cannot believe how bad this was I am enraged",
        "The characters were awful and I felt manipulated throughout",
        "Utterly infuriating film I am still angry hours later",
        "What a disaster I am furious I paid for this garbage",
        "Offensive tone-deaf and deeply annoying movie",
        "I am enraged by the plot holes and lazy writing",
        "This movie disrespected its audience I am outraged",
        "Horrible acting and a disgusting storyline I am livid",
        "The movie was not good at all it was terrible",
        "This is the worst movie I have ever seen in my life",
        "Do not watch this film it is absolutely terrible",
        "I am so disappointed and angry about this movie",
        "Nothing about this movie worked it was all bad",
        "A complete failure on every level",
        "I wasted two hours of my life on this nonsense",
        "The plot makes no sense and the acting is horrible",
        "I would not recommend this to anyone avoid it",
        "This movie is an insult to cinema",
        # --- Courtes (FR) ---
        "mauvais film", "terrible", "horrible", "nul", "le pire",
        "tres mauvais", "je deteste", "degoutant", "catastrophe", "desastre",
        "une horreur", "inacceptable", "lamentable", "deplorable", "honteux",
        "pas bon", "vraiment nul", "affreux", "execrable", "insupportable",
        # --- Moyennes (FR) ---
        "Ce film est une perte de temps totale je suis furieux",
        "Le realisateur a tout gache je suis tellement en colere",
        "Scenario terrible offensant et profondement frustrant a regarder",
        "Je deteste ce film quelle insulte pour le public",
        "Ce film m'a rendu incroyablement en colere mauvaise narration",
        "Je suis sorti du cinema indigne gaspillage d'argent degoutant",
        "Ennuyeux et exasperant le pire film que j'aie vu",
        "Les personnages etaient affreux et je me suis senti manipule",
        "Film totalement exasperant je suis encore en colere",
        "Ce film a manque de respect a son public je suis outre",
        "Jeu d'acteurs horrible et une histoire degoutante",
        "Le film n'est pas bon du tout il est terrible",
        "Ce film est le pire que j'aie jamais vu",
        "Ne regardez pas ce film il est absolument terrible",
        "Rien dans ce film ne fonctionne tout est mauvais",
        "the movie is awfull",
    ],

    "neutral": [
        "the movie is normal", "it's okay", "normal movie", "average",
        "nothing special", "so so", "it was fine", "not bad not good",
        "just okay", "decent", "regular", "standard", "it's alright",
        "film normal", "moyen", "correct", "pas mal", "ni bon ni mauvais",
        "sans plus", "film quelconque", "ordinaire", "ca va", "ok",
        "the film is normal", "normal film", "it is ok", "it is okay",
    ],

    "sadness": [
        # --- Courtes (EN) ---
        "so sad", "made me cry", "heartbreaking", "depressing", "devastating",
        "I cried", "very sad", "tragic", "sorrowful", "melancholic",
        "tears", "grief", "miserable", "gloomy", "painful to watch",
        "emotionally draining", "touching", "moving", "I wept", "hopeless",
        "broke my heart", "soul crushing", "tearful", "woeful", "pitiful",
        "deeply sad", "I felt empty", "heavy heart", "gut wrenching",
        "makes you cry", "tear jerker", "bittersweet ending",
        # --- Moyennes (EN) ---
        "This movie made me cry so much incredibly sad story",
        "I felt so heartbroken after watching this film",
        "Such a depressing movie I felt hopeless and melancholic",
        "The ending devastated me I could not stop crying",
        "A deeply moving and sorrowful film that touched my soul",
        "I felt a deep sadness throughout the entire movie",
        "This film left me feeling empty and grief-stricken",
        "So tragic and emotional I wept for hours after",
        "The characters suffered so much I felt their pain deeply",
        "A tearjerker from start to finish utterly heartbreaking",
        "This movie broke my heart into a thousand pieces",
        "I was in tears the entire time watching this sad film",
        "The tragedy in this movie left me feeling devastated",
        "Such a gloomy and melancholy film it made me despair",
        "I felt a profound sadness after watching this moving story",
        "The story was so emotional and sad I could not handle it",
        "What a devastating and heartbreaking ending to the film",
        "I have never cried this much watching a movie before",
        "The sadness in this film was overwhelming and powerful",
        "This movie left a deep emotional mark on me",
        "A truly heartwrenching story that will make you cry",
        "The characters journey was painful and full of sorrow",
        "I felt depressed after watching this tragic movie",
        "A beautiful but deeply sad and melancholic film",
        "This movie captures the pain of loss perfectly",
        # --- Courtes (FR) ---
        "si triste", "j'ai pleure", "dechirante", "deprimant", "devastant",
        "emouvant", "melancolique", "tragique", "douloureux", "sombre",
        "larmes", "chagrin", "desespoir", "le coeur brise", "accablant",
        "triste histoire", "tres emouvant", "fait pleurer", "poignant",
        # --- Moyennes (FR) ---
        "Ce film m'a fait pleurer tellement une histoire incroyablement triste",
        "Je me suis senti le coeur brise apres avoir regarde ce film",
        "Film si deprimant je me suis senti sans espoir et melancolique",
        "La fin m'a devaste je n'arretais pas de pleurer",
        "Un film profondement emouvant et melancolique qui a touche mon ame",
        "J'ai ressenti une profonde tristesse tout au long du film",
        "Ce film m'a laisse vide et accable de chagrin",
        "Si tragique et emotionnel j'ai pleure pendant des heures",
        "Les personnages ont tellement souffert j'ai profondement ressenti leur douleur",
        "Un film qui fait pleurer du debut a la fin totalement dechirant",
        "Ce film m'a brise le coeur en mille morceaux",
        "J'etais en larmes pendant tout le film tellement il etait triste",
        "La tragedie dans ce film m'a laisse devaste",
        "Film si sombre et melancolique ca m'a desespere",
        "J'ai ressenti une profonde tristesse apres avoir vu cette histoire emouvante",
    ],

    "fear": [
        # --- Courtes (EN) ---
        "terrifying", "scary", "frightening", "creepy", "haunting",
        "nightmare", "horrifying", "chilling", "disturbing", "unsettling",
        "I was scared", "so scary", "gave me nightmares", "spine chilling",
        "bone chilling", "blood curdling", "petrifying", "dreadful",
        "I was terrified", "could not sleep", "nerve wracking", "tense",
        "anxious", "fearful", "alarming", "sinister", "menacing",
        "I was shaking", "goosebumps", "made me jump", "dark and creepy",
        "pure horror", "deeply unsettling", "paranoid after watching",
        # --- Moyennes (EN) ---
        "This horror movie terrified me I was shaking throughout",
        "I was scared out of my mind watching this film",
        "So frightening and disturbing I had nightmares afterwards",
        "The tension was unbearable I was paralysed with fear",
        "Absolutely terrifying I could not sleep after watching this",
        "The suspense made me anxious and deeply unsettled",
        "I was on edge the entire time this film is truly scary",
        "The jump scares gave me a heart attack I was petrified",
        "So haunting and creepy I felt dread throughout the film",
        "I was hiding behind my hands for most of this film",
        "This psychological thriller left me feeling paranoid and afraid",
        "The atmosphere was so ominous I was filled with dread",
        "Genuinely terrifying I felt fear crawling down my spine",
        "The horror was relentless I was genuinely frightened",
        "So nerve-wracking and tense I was trembling with fear",
        "This movie kept me up all night in fear",
        "I have never been so scared watching a movie before",
        "The dark atmosphere and sounds made it truly terrifying",
        "Every scene filled me with dread and anxiety",
        "A genuinely horrifying experience from start to finish",
        "The villain was so creepy it made my skin crawl",
        "I was too afraid to look at the screen",
        "This film is not for the faint of heart truly scary",
        "The suspense was killing me I was so anxious",
        "A deeply unsettling and psychologically disturbing movie",
        # --- Courtes (FR) ---
        "terrifiant", "effrayant", "flippant", "cauchemar", "angoissant",
        "j'ai eu peur", "fait peur", "horrible", "glauque", "sinistre",
        "stressant", "inquietant", "menaçant", "petrifiant", "affolant",
        "j'etais terrfie", "je tremblais", "cauchemardesque", "lugubre",
        # --- Moyennes (FR) ---
        "Ce film d'horreur m'a terrifie je tremblais tout au long",
        "J'etais mort de peur en regardant ce film",
        "Si effrayant et troublant j'ai eu des cauchemars apres",
        "La tension etait insupportable j'etais paralyse par la peur",
        "Absolument terrifiant je n'arrivais pas a dormir apres",
        "Le suspense m'a rendu anxieux et profondement trouble",
        "J'etais sur les nerfs tout le temps ce film est vraiment effrayant",
        "Les jump scares m'ont donne une crise cardiaque j'etais petrifie",
        "Si hante et sinistre j'ai ressenti de l'angoisse tout au long",
        "Ce thriller psychologique m'a laisse paranoiaque et effraye",
        "L'atmosphere etait si menacante j'etais rempli d'effroi",
        "Vraiment terrifiant j'ai senti la peur courir dans mon dos",
        "L'horreur etait implacable j'ai vraiment eu peur",
        "Si stressant et tendu je tremblais de peur",
        "Un film vraiment glauque et angoissant du debut a la fin",
    ],

    "surprise": [
        # --- Courtes (EN) ---
        "unexpected", "shocking", "plot twist", "did not see that coming",
        "mind blowing", "jaw dropping", "unbelievable", "astonishing",
        "I was shocked", "stunned", "speechless", "what a twist",
        "unpredictable", "caught me off guard", "blown away",
        "I gasped", "what just happened", "unforeseeable", "startling",
        "incredible twist", "totally unexpected", "I was amazed",
        "never expected that", "gobsmacked", "flabbergasted",
        "that ending though", "what a reveal", "so unpredictable",
        "completely blindsided", "left me speechless",
        # --- Moyennes (EN) ---
        "I was completely shocked by the plot twist did not see that coming",
        "The ending totally surprised me I was left speechless",
        "I could not believe what I just watched so unexpected",
        "The film took me completely off guard with its twists",
        "Absolutely astonishing I was blown away by the revelations",
        "The shocking twist left my jaw on the floor",
        "I had no idea this would happen what a stunning surprise",
        "The unexpected ending left me in total disbelief",
        "Nothing could have prepared me for that final scene",
        "I was gobsmacked by the reveal completely unforeseen",
        "Such an unpredictable film I was constantly surprised",
        "I gasped out loud at the plot twist truly astonishing",
        "The surprise ending was totally mind-blowing",
        "I was stunned and amazed by the direction this film took",
        "The unexpected narrative turns kept me in constant shock",
        "I never would have guessed that twist ending",
        "The story went in a completely different direction than expected",
        "My mind was blown by the final revelation",
        "This film defied all my expectations in the best way",
        "I was not prepared for what this movie had in store",
        "The plot twists were genuinely surprising and well crafted",
        "I sat there in shock after the credits rolled",
        "This movie keeps you guessing until the very last moment",
        "What an incredible and unexpected journey this film was",
        "The revelations in this movie left me absolutely stunned",
        # --- Courtes (FR) ---
        "surprenant", "choquant", "inattendu", "incroyable rebondissement",
        "je suis choque", "bouche bee", "stupefiant", "epoustouflant",
        "impensable", "imprevisible", "sans voix", "abasourdi",
        "je ne m'y attendais pas", "quelle surprise", "renversant",
        "inimaginable", "saisissant", "deconcertant", "quel retournement",
        # --- Moyennes (FR) ---
        "J'ai ete completement choque par le rebondissement",
        "La fin m'a totalement surpris je suis reste sans voix",
        "Je n'arrivais pas a croire ce que je venais de regarder",
        "Le film m'a totalement pris de court avec ses rebondissements",
        "Absolument stupefiant j'ai ete souffle par les revelations",
        "Le rebondissement choquant m'a laisse bouche bee",
        "La fin inattendue m'a laisse dans une incredulite totale",
        "Rien n'aurait pu me preparer a cette scene finale",
        "J'ai ete stupefait par la revelation totalement imprevisible",
        "Film si imprevisible j'ai ete constamment surpris",
        "J'ai pousse un cri de surprise au rebondissement",
        "La fin surprise etait totalement epoustouflante",
        "J'ai ete abasourdi par la direction que ce film a prise",
        "Les tournants narratifs inattendus m'ont garde dans un choc constant",
        "Quel retournement de situation je ne m'y attendais vraiment pas",
    ],
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zàâäéèêëîïôùûüç\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES (CSV ou Manuel)
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    csv_path = 'emotions_dataset.csv'
    X, y = [], []
    
    # 1. Charger les SAMPLES intégrés d'abord
    print("[INFO] Chargement du dataset intégré au script...")
    for emotion, phrases in SAMPLES.items():
        for phrase in phrases:
            X.append(clean_text(phrase))
            y.append(emotion)
            
    # 2. Ajouter les données du CSV s'il existe
    if os.path.exists(csv_path):
        print(f"[INFO] Ajout des données depuis le dataset externe : {csv_path}")
        df = pd.read_csv(csv_path)
        if 'text' in df.columns and 'emotion' in df.columns:
            for _, row in df.iterrows():
                X.append(clean_text(row['text']))
                y.append(row['emotion'])
        print(f"[INFO] {len(df)} phrases supplémentaires ajoutées.")
    
    return X, y

X, y = load_data()

print(f"Total des données pour l'entraînement : {len(X)} phrases")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ══════════════════════════════════════════════════════════════════════════════
# ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════════════
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),      # unigrams + bigrams + trigrams
    max_features=15000,
    sublinear_tf=True,
    min_df=1,
)
model = LinearSVC(C=0.8, max_iter=5000, class_weight='balanced')

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)
model.fit(X_train_vec, y_train)

# ══════════════════════════════════════════════════════════════════════════════
# ÉVALUATION
# ══════════════════════════════════════════════════════════════════════════════
y_pred = model.predict(X_test_vec)
print("\n" + "="*60)
print("  RAPPORT DE CLASSIFICATION - MODELE MULTI-EMOTIONS")
print("="*60)
print(classification_report(y_test, y_pred))

# Matrice de confusion (texte)
labels = ["joy", "anger", "sadness", "fear", "surprise", "neutral"]
cm = confusion_matrix(y_test, y_pred, labels=labels)
print("\nMatrice de confusion :")
print("         ", "  ".join(f"{l:8s}" for l in labels))
for i, row in enumerate(cm):
    print(f"{labels[i]:10s}", "  ".join(f"{v:8d}" for v in row))

# ══════════════════════════════════════════════════════════════════════════════
# SAUVEGARDE DES MODÈLES
# ══════════════════════════════════════════════════════════════════════════════
joblib.dump(model,      'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("\n[OK] Modele sauvegarde     : sentiment_model.pkl")
print("[OK] Vectoriseur sauvegarde: vectorizer.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# TEST RAPIDE - incluant les phrases de l'utilisateur
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  TESTS RAPIDES")
print("="*60)
tests = [
    # Tests de l'utilisateur qui échouaient avant
    "the movie is normal",
    "the movie is not good",
    "the movie is awful",
    "the movie is bad",
    "the movie is so bad",
    "the movie is so good",
    # Autres tests
    "I loved this movie, it was amazing and so much fun!",
    "This film made me so angry, what a waste of time",
    "I cried so much, such a sad and heartbreaking story",
    "This horror movie terrified me, I could not sleep",
    "The plot twist shocked me completely, I did not see it coming",
    "J'ai adore ce film, c'etait magnifique!",
    "Ce film m'a mis en colere, quelle perte de temps",
    "J'ai tellement pleure, histoire si triste et dechirante",
    "Ce film d'horreur m'a terrifie",
    "Le rebondissement m'a choque",
]

EMOTION_LABELS = {
    "joy":      "Content",
    "anger":    "En colere",
    "sadness":  "Triste",
    "fear":     "Peur",
    "surprise": "Surpris",
    "neutral":  "Neutre",
}

for t in tests:
    vec = vectorizer.transform([clean_text(t)])
    pred = model.predict(vec)[0]
    label = EMOTION_LABELS[pred]
    print(f"  [{label:12s}] {t[:70]}")

print("\n[OK] Entrainement termine avec succes !")
print("     Relancez maintenant : streamlit run app.py")
