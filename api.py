import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Initialisation de FastAPI avec l'identité de BlaDino
app = FastAPI(title="BlaDino - L'Assistant KenzBladi", version="1.4")

# Configuration du CORS pour permettre à votre future maquette de communiquer avec le serveur Python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("⏳ Connexion à la base de connaissances KenzBladi...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
print("✅ Base vectorielle chargée. BlaDino est prêt à répondre !")

# Structure de la question que la maquette va envoyer au format JSON
class FormatQuestion(BaseModel):
    question: str

# La route principale (l'URL) que le chatbot va utiliser
@app.post("/api/chat")
def repondre_au_client(donnees: FormatQuestion):
    try:
        question_utilisateur = donnees.question.lower()
        
        resultats = db.similarity_search(question_utilisateur, k=5)
        
        if len(resultats) > 0:
            produits_trouves = []
            for res in resultats:
                contenu = res.page_content
                if "Produit/Patrimoine : " in contenu:
                    nom = contenu.split("Produit/Patrimoine : ")[1].split(".")[0].strip()
                    carac = ""
                    if "Caractéristiques : " in contenu:
                        carac = contenu.split("Caractéristiques : ")[1].strip()
                    produits_trouves.append({"nom": nom, "carac": carac})
            
            if produits_trouves:
                reponse = f"Bonjour ! Je suis BlaDino 🌿\n\nVoici ce que j'ai trouvé :\n\n"
                for p in produits_trouves[:4]:
                    reponse += f"✅ **{p['nom']}**"
                    if p['carac']:
                        reponse += f"\n{p['carac']}"
                    reponse += "\n\n"
            else:
                reponse = "Bonjour ! Je n'ai pas trouvé ce produit dans notre catalogue."
        else:
            reponse = "Bonjour ! Je n'ai pas trouvé ce produit dans notre catalogue KenzBladi."
            
        return {"reponse": reponse}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))








@app.get("/")
def index():
    return {"status": "En ligne", "application": "BlaDino Chatbot Server"}