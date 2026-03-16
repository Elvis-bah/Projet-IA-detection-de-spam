# 📧 Détection de Spam via Machine Learning (Projet-IA-detection-de-spam)

## 📋 Présentation du Projet
Ce projet utilise des techniques de **Traitement du Langage Naturel (NLP)** et de **Machine Learning** pour classifier automatiquement les messages en tant que "Spam" ou "Ham" (messages légitimes). L'objectif est de fournir une solution efficace pour filtrer les communications indésirables et améliorer la cybersécurité.

## 🛠️ Stack Technique
* **Langage :** Python 🐍
* **Bibliothèques :** * `Pandas` & `NumPy` (Manipulation de données)
    * `Scikit-learn` (SVM, Naive Bayes, Random Forest)
    * `NLTK` ou `Spacy` (Prétraitement du texte)
* **Outils :** Jupyter Notebook, Git, WSL/Ubuntu

## 🚀 Fonctionnalités du Projet
Le projet suit un pipeline complet de Data Science :
1. **Prétraitement du texte :** Nettoyage (minuscules, suppression de la ponctuation), suppression des mots vides (stopwords) et lemmatisation.
2. **Vectorisation :** Conversion du texte en vecteurs numériques grâce à `CountVectorizer` ou `TF-IDF`.
3. **Modélisation :** Entraînement de modèles de classification supervisée.
4. **Évaluation :** Analyse des performances via la matrice de confusion et le score de précision (Accuracy).

## 📂 Structure du Dépôt
```bash
.
├── docs/                       # Documentation et ressources additionnelles
├── detection_spam_ham.ipynb    # Notebook principal (Analyse et Modèles)
├── nettoyage_donnees_UA2.py     # Script Python pour le nettoyage automatisé
└── README.md                   # Documentation du projet
