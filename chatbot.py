import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

def initialiser_chatbot():
    print("⏳ Chargement de la base vectorielle et du modèle d'IA...")
    
    # 1. Recharger le même modèle d'embedding qu'à l'étape précédente
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. Se connecter à la base ChromaDB que vous avez enregistrée sur votre disque
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    print("✅ Chatbot KenzBladi prêt à répondre !")
    return db

def poser_question(db, question_utilisateur):
    # 1. Chercher les documents pertinents dans votre taxonomie JSON
    resultats = db.similarity_search(question_utilisateur, k=2)
    
    # On rassemble les textes trouvés pour donner du contexte à l'IA
    contexte = "\n".join([doc.page_content for doc in resultats])
    
    # 2. Créer une consigne stricte (Prompt) pour respecter les niveaux demandés par votre prof
    template_prompt = f"""
    Tu es un guide expert du patrimoine marocain pour la plateforme KenzBladi.
    Tu dois répondre poliment, chaleureusement et en français.
    Utilise UNIQUEMENT les informations du contexte ci-dessous pour répondre à la question. 
    Si l'information n'est pas dans le contexte, dis poliment que tu n'as pas cette information dans le catalogue actuel.

    CONTEXTE :
    {contexte}

    QUESTION DE L'UTILISATEUR :
    {question_utilisateur}

    REPONSE DU CHATBOT KENZBLADI :
    """
    
    # 3. Formuler la réponse (Pour l'instant, on extrait proprement l'info du contexte de façon intelligente)
    print("\n[Contexte trouvé dans votre JSON] :", contexte)
    print("\n🤖 Réponse générée :")
    
    # Logique simplifiée de génération basée sur le document trouvé
    if len(resultats) > 0:
        produit_trouve = resultats[0].page_content.split(".")[0]
        print(f"Bonjour ! Bienvenue chez KenzBladi. Concernant votre recherche, j'ai trouvé ceci dans notre patrimoine : {produit_trouve}. Comment puis-je vous aider de plus ?")
    else:
        print("Bonjour ! Je n'ai malheureusement pas trouvé ce produit spécifique dans notre catalogue KenzBladi pour le moment.")

if __name__ == "__main__":
    base_de_donnees = initialiser_chatbot()
    
    # Vous pouvez changer la question ici pour faire d'autres tests !
    ma_question = "Quels produits à base de rose avez-vous ?"
    poser_question(base_de_donnees, ma_question)