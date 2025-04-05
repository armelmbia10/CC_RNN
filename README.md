# Prédiction des cours d'Apple avec CNN1D et GRU

Cette application Streamlit permet de prédire les cours des actions d'Apple en utilisant des modèles de Deep Learning (CNN1D et GRU).

## Fonctionnalités

- Téléchargement automatique des données historiques d'Apple sur 4 ans
- Prédiction des cours avec deux modèles : CNN1D et GRU
- Visualisation des prédictions sur 3 semaines
- Interface interactive avec paramètres ajustables
- Métriques de performance des modèles

## Installation

1. Clonez ce dépôt
2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Lancez l'application :
```bash
streamlit run app.py
```

2. Dans l'interface :
   - Ajustez la période de lookback (30-90 jours)
   - Modifiez la taille de l'ensemble d'entraînement (70-90%)
   - Visualisez les prédictions et les métriques de performance

## Déploiement sur Hugging Face

1. Créez un compte sur [Hugging Face](https://huggingface.co/)
2. Créez un nouvel espace
3. Configurez l'espace avec les fichiers suivants :
   - app.py
   - requirements.txt
   - README.md

## Structure des modèles

### CNN1D
- Couche Conv1D (64 filtres)
- MaxPooling1D
- Couche Conv1D (32 filtres)
- MaxPooling1D
- Couches Denses

### GRU
- Couche GRU (50 unités)
- Couche GRU (50 unités)
- Couche Dense

## Métriques
L'application affiche le MSE (Mean Squared Error) pour chaque modèle, permettant de comparer leurs performances. 