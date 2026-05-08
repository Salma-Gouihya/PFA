import sys
import os

# Chemins
user_site = r"C:\Users\poste\AppData\Roaming\Python\Python313\site-packages"
broken_site = r"C:\Python313\Lib\site-packages"

# Nettoyage agressif de sys.path
# 1. On enlève le dossier cassé s'il existe
if broken_site in sys.path:
    sys.path.remove(broken_site)

# 2. On force le dossier utilisateur au tout début
if user_site in sys.path:
    sys.path.remove(user_site)
sys.path.insert(0, user_site)

# 3. On vérifie s'il n'y a pas un dossier 'streamlit' local qui gêne
if os.path.exists("streamlit"):
    print("ATTENTION : Un dossier nommé 'streamlit' existe ici et crée un conflit !")

try:
    from streamlit.web import cli as stcli
    import streamlit
    print(f"SUCCÈS ! Streamlit chargé depuis : {streamlit.__file__}")
    
    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(stcli.main())
except Exception as e:
    print(f"Échec critique : {e}")
    print("\nOrdre de recherche final :")
    for p in sys.path:
        print(f"- {p}")
