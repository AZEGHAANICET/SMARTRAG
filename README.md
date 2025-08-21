# 🚀 SmartRAG 🤖📚

*"Agentic RAG pour recherche intelligente et contextuelle dans vos documents et sources externes"*  

---

## 🎯 Description

**SmartRAG** est un agent intelligent basé sur **RAG** (*Retrieval-Augmented Generation*) qui permet de :  

- 💡 Interroger vos documents internes (`.txt`) et récupérer des informations pertinentes.  
- 🌐 Interroger des sources externes comme **Wikipedia** et **ArXiv**.  
- 🧠 Générer des réponses contextuelles en combinant les connaissances internes et externes.  
- 🖥️ Interface interactive via **Streamlit** pour une utilisation facile.

L'agent **décide automatiquement** quel outil utiliser et injecte le contexte récupéré pour produire une réponse précise.

---

## ⚙️ Fonctionnalités principales

1. **Recherche intelligente dans vos fichiers**  
   - Documents internes transformés en vecteurs via **FAISS** et embeddings OpenAI.  
   - Recherche contextuelle avec split automatique en chunks.  

2. **Intégration de sources externes**  
   - Wikipedia pour des connaissances générales.  
   - ArXiv pour les derniers articles scientifiques.  

3. **Agentic RAG**  
   - L’agent choisit automatiquement le meilleur outil pour répondre à la question.  
   - Le contexte récupéré est injecté dans le LLM pour générer une réponse complète et cohérente.  

4. **Interface interactive avec Streamlit**  
   - Champ de saisie pour la question.  
   - Bouton "Search" pour obtenir la réponse instantanément.  

---

## 🛠️ Installation

1. Cloner le projet :  
```bash
git clone https://github.com/votre_compte/SmartRAG.git
cd SmartRAG

# STRUCTURE DU PROJET 

SmartRAG/
│
├─ app.py                # Code principal Streamlit + Agentic RAG
├─ research_notes.txt    # Vos notes internes
├─ sample_docs.txt       # Documents supplémentaires
├─ requirements.txt      # Dépendances Python
└─ README.md             # Ce fichier
