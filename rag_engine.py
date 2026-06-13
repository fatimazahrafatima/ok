import json
import os
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def charger_donnees_kenzbladi():
    documents = []
    chemin_fichier = r"C:\Users\DELL\Desktop\KenzBladi_Chatbot\data\taxonomie_clean.json"
    
    with open(chemin_fichier, "r", encoding="utf-8") as f:
        taxonomies = json.load(f)
    
    # Créer un index par ID
    index = {str(item["_id"]["$oid"]): item for item in taxonomies}
    
    for item in taxonomies:
        translations = item.get("translations", [{}])
        designation = translations[0].get("designation", "Inconnu")
        carac = item.get("metadata", {}).get("caracteristiquesRaw", "")
        
        # Récupérer le nom du parent si existant
        parent_nom = ""
        if item.get("parent"):
            parent_id = str(item["parent"]["$oid"])
            if parent_id in index:
                parent_translations = index[parent_id].get("translations", [{}])
                parent_nom = parent_translations[0].get("designation", "")
        
        # Construire le texte enrichi
        if parent_nom:
            contenu_texte = f"Produit/Patrimoine : {parent_nom} - {designation}. Caractéristiques : {carac}"
        else:
            contenu_texte = f"Produit/Patrimoine : {designation}. Caractéristiques : {carac}"
        
        doc = Document(
            page_content=contenu_texte,
            metadata={"source": "taxonomie", "code": item.get("code")}
        )
        documents.append(doc)
    
    print(f"✅ {len(documents)} documents chargés en mémoire.")
    return documents

def creer_base_vectorielle(documents):
    print("⏳ Initialisation du modèle d'IA (Embedding)... Merci de patienter...")
    
    # On utilise un modèle léger et performant pour le français/multilingue
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("⏳ Création de la base de données vectorielle Chroma... (Cela peut prendre 1 à 2 minutes la première fois)")
    
    # Création de la base de données dans un dossier nommé "chroma_db"
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("✅ Base de données vectorielle créée avec succès et enregistrée dans './chroma_db' !")
    return vector_store

if __name__ == "__main__":
    # 1. Charger les textes
    docs = charger_donnees_kenzbladi()
    
    # 2. Les transformer en vecteurs
    db = creer_base_vectorielle(docs)
    
    # 3. TEST : On simule une recherche de l'utilisateur
    requete = "Je cherche de l'huile de pépins de figue de barbarie"
    print(f"\n🔍 Test de recherche pour : '{requete}'")
    
    resultats = db.similarity_search(requete, k=2) # On demande les 2 meilleurs résultats
    
    print("\n🤖 Résultats trouvés par l'IA :")
    for i, res in enumerate(resultats):
        print(f"\n--- Résultat {i+1} ---")
        print(res.page_content)