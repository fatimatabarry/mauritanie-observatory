# 🌍 Mauritania Open Data Observatory

Agent IA capable de répondre à des questions sur la Mauritanie,
construit avec LangGraph, RAG, ChromaDB et Groq.

## 🚀 Application en ligne
👉 https://mauritanie-observatory.streamlit.app

## 🛠️ Technologies utilisées

| Outil | Rôle |
|---|---|
| LangGraph | Orchestration des agents |
| RAG + ChromaDB | Recherche sémantique dans les données |
| Groq LLM | Génération de réponses llama3-8b |
| HuggingFace Embeddings | Conversion texte en vecteurs |
| Streamlit | Interface utilisateur interactive |

## 🏗️ Architecture

    Question utilisateur
          ↓
    Agent RAG → ChromaDB (recherche documents pertinents)
          ↓
    Agent Réponse → Groq LLM (génère la réponse)
          ↓
    Réponse en français basée sur les données

## 🤖 Agents LangGraph

- RAG Agent → cherche les documents pertinents dans ChromaDB
- Response Agent → génère la réponse avec le LLM

## 📊 Données disponibles

- Population et démographie
- PIB et économie
- Alphabétisation et éducation
- Espérance de vie et santé
- Ressources naturelles (fer, cuivre, pétrole, gaz)
- Géographie et villes principales
- Politique et institutions

## ▶️ Lancer en local

    pip install -r requirements.txt
    streamlit run app.py

## 📁 Structure du projet

    mauritanie-observatory/
    ├── app.py            # Interface Streamlit + Agent IA
    ├── requirements.txt  # Dépendances Python
    └── README.md         # Documentation

## 👩‍💻 Auteur
Fatimata Barry — TD7 Mauritania Open Data Observatory
