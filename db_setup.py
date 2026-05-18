import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import json
import os
from passlib.hash import pbkdf2_sha256

# Configuration
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "user": "postgres",
    "password": "Goui3006",
    "dbname": "postgres"  # On commence par se connecter a la base par defaut
}
TARGET_DB = "Goui3006"

def setup_database():
    try:
        # 1. Connexion pour creer la base de donnees si elle n'existe pas
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        cur.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{TARGET_DB}'")
        exists = cur.fetchone()
        if not exists:
            print(f"Creation de la base de donnees '{TARGET_DB}'...")
            cur.execute(f"CREATE DATABASE \"{TARGET_DB}\"")
        
        cur.close()
        conn.close()
        
        # 2. Connexion a la base cible pour creer les tables
        print(f"Connexion a la base '{TARGET_DB}'...")
        conn = psycopg2.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            dbname=TARGET_DB
        )
        cur = conn.cursor()
        
        # 3. Tables
        print("Creation des tables...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role VARCHAR(20) DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS comments (
                id SERIAL PRIMARY KEY,
                film_id INTEGER,
                film_title TEXT,
                text TEXT,
                sentiment TEXT,
                style TEXT,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL
            )
        """)
        
        # 4. Admin par defaut
        cur.execute("SELECT id FROM users WHERE username = 'admin'")
        if not cur.fetchone():
            print("Creation du compte admin...")
            hashed_pw = pbkdf2_sha256.hash("admin123")
            cur.execute("INSERT INTO users (username, password, role) VALUES (%s, %s, %s)", 
                        ('admin', hashed_pw, 'admin'))

        # 5. Migration JSON
        if os.path.exists("comments.json"):
            print("Migration des donnees depuis comments.json...")
            try:
                with open("comments.json", "r", encoding="utf-8") as f:
                    comments = json.load(f)
            except:
                with open("comments.json", "r", encoding="latin-1") as f:
                    comments = json.load(f)
            
            # Vider la table avant migration pour eviter les doublons lors des tests
            cur.execute("TRUNCATE TABLE comments CASCADE")
            
            for c in comments:
                cur.execute("""
                    INSERT INTO comments (film_id, film_title, text, sentiment, style, date)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (c['film_id'], c['film_title'], c['text'], c['sentiment'], c['style'], c['date']))
            print(f"{len(comments)} commentaires migres.")
            
        conn.commit()
        cur.close()
        conn.close()
        print("[OK] PostgreSQL est pret !")
        
    except Exception as e:
        # On evite d'afficher directement 'e' s'il contient des caracteres speciaux
        print(f"[ERREUR] Un probleme est survenu lors de la configuration.")
        # print(str(e)) # Optionnel si on veut deboguer en local

if __name__ == "__main__":
    setup_database()
