import pandas as pd
import requests
import os

def download_and_map():
    url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/train.tsv"
    csv_output = "emotions_dataset.csv"
    
    print(f"Téléchargement du dataset GoEmotions (Google Research)...")
    try:
        response = requests.get(url)
        with open("raw_data.tsv", "wb") as f:
            f.write(response.content)
            
        # Charger le TSV
        df = pd.read_csv("raw_data.tsv", sep='\t', header=None, names=['text', 'labels', 'id'])
        
        # Mapping des 27 émotions vers vos 6 catégories
        # 0:admiration, 1:amusement, 2:anger, 3:annoyance, 4:approval, 13:excitement, 17:joy, 18:love, 20:optimism, 21:pride, 23:relief -> JOY
        # 2:anger, 3:annoyance, 10:disapproval, 11:disgust -> ANGER
        # 9:disappointment, 16:grief, 24:remorse, 25:sadness -> SADNESS
        # 14:fear, 19:nervousness -> FEAR
        # 26:surprise -> SURPRISE
        # 27:neutral -> NEUTRAL
        
        mapping = {
            "joy": [0, 1, 4, 13, 15, 17, 18, 20, 21, 23],
            "anger": [2, 3, 10, 11],
            "sadness": [9, 16, 24, 25],
            "fear": [14, 19],
            "surprise": [26],
            "neutral": [27, 5, 6, 7, 8, 12, 22]
        }
        
        final_data = []
        
        for _, row in df.iterrows():
            text = row['text']
            labels = [int(l) for l in str(row['labels']).split(',')]
            
            mapped_emotion = None
            for emotion, source_labels in mapping.items():
                if any(l in source_labels for l in labels):
                    mapped_emotion = emotion
                    break
            
            if mapped_emotion:
                final_data.append({"text": text, "emotion": mapped_emotion})
        
        # Créer le nouveau CSV
        new_df = pd.DataFrame(final_data)
        
        # Ajouter quelques exemples en français manuellement pour garder le côté bilingue
        french_data = [
            {"text": "ce film est magnifique", "emotion": "joy"},
            {"text": "j'ai détesté ce film", "emotion": "anger"},
            {"text": "quelle déception", "emotion": "sadness"},
            {"text": "j'ai eu très peur", "emotion": "fear"},
            {"text": "le dénouement m'a choqué", "emotion": "surprise"},
            {"text": "un film assez moyen", "emotion": "neutral"}
        ]
        new_df = pd.concat([new_df, pd.DataFrame(french_data)], ignore_index=True)
        
        new_df.to_csv(csv_output, index=False)
        print(f"Succès ! {len(new_df)} phrases professionnelles enregistrées dans {csv_output}")
        
        # Nettoyage
        if os.path.exists("raw_data.tsv"): os.remove("raw_data.tsv")
        
    except Exception as e:
        print(f"Erreur lors du téléchargement : {e}")

if __name__ == "__main__":
    download_and_map()
