# Diagrammes UML - Projet CineStream

Ce fichier contient les diagrammes UML du projet CineStream (Analyse Multimodale de Sentiments). Les diagrammes sont au format Mermaid.

## 1. Diagramme de Cas d'Utilisation
Ce diagramme illustre les interactions entre les acteurs (Client, Administrateur) et les fonctionnalités du système.

```mermaid
useCaseDiagram
    actor "Utilisateur (Client)" as Client
    actor "Administrateur" as Admin

    package "Système CineStream" {
        usecase "Parcourir le catalogue" as UC1
        usecase "Voir les détails d'un film" as UC2
        usecase "Laisser un avis écrit" as UC3
        usecase "Enregistrer un avis vocal" as UC4
        usecase "Analyse de sentiment auto" as UC5
        usecase "Consulter les statistiques" as UC6
        usecase "Gérer/Réinitialiser les données" as UC7
    }

    Client --> UC1
    Client --> UC2
    Client --> UC3
    Client --> UC4
    
    UC3 ..> UC5 : <<include>>
    UC4 ..> UC5 : <<include>>

    Admin --> UC6
    Admin --> UC7
```

---

## 2. Diagramme de Classe
Ce diagramme décrit la structure statique du code, incluant les entités et les relations logiques.

```mermaid
classDiagram
    class App {
        +session_state : dict
        +main()
        +render_home()
        +render_detail(film_id)
        +render_admin()
    }

    class Movie {
        +int id
        +string title
        +string genre
        +float rating
        +string backdrop
        +string poster
        +string synopsis
    }

    class Comment {
        +int id
        +int film_id
        +string text
        +string sentiment
        +string style
        +datetime date
    }

    class SentimentModel {
        +model : SVM
        +vectorizer : TF-IDF
        +clean_text(text)
        +predict(text)
    }

    class VoiceProcessor {
        +recognizer : sr.Recognizer
        +transcribe(audio_data)
    }

    class DataManager {
        +FILE_PATH : string
        +load_comments()
        +save_comment(data)
        +reset_data()
    }

    App --> Movie : "affiche"
    App --> Comment : "crée/lit"
    App --> SentimentModel : "utilise"
    App --> VoiceProcessor : "utilise"
    DataManager -- App : "persistance"
```

---

## 3. Diagramme de Séquence (Avis Vocal)
Ce diagramme montre l'interaction dynamique entre les composants lors de l'enregistrement d'un avis vocal.

```mermaid
sequenceDiagram
    participant U as Utilisateur
    participant S as Streamlit (UI)
    participant V as VoiceProcessor
    participant G as Google Speech API
    participant AI as SentimentModel
    participant D as DataManager (JSON)

    U->>S: Clique sur enregistrer (st.audio_input)
    S-->>U: Capture l'audio (WAV)
    U->>S: Arrête l'enregistrement
    S->>V: Envoie le flux audio (Bytes)
    V->>G: Envoie l'audio pour transcription
    G-->>V: Retourne le texte (String)
    V->>S: Affiche la transcription
    S->>AI: Envoie le texte pour analyse
    AI-->>S: Retourne le Sentiment (Positif/Négatif)
    S->>D: Demande de sauvegarde (save_comment)
    D-->>S: Confirmation (JSON mis à jour)
    S-->>U: Affiche succès + Sentiment
```
