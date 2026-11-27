# EuroMillions — Analyse statistique

Application [Streamlit](https://streamlit.io) qui analyse l'historique officiel des tirages EuroMillions (FDJ) et génère des grilles.

> **À lire avant tout.** Chaque tirage est indépendant et chaque combinaison a exactement 1 chance sur 139 838 160 de sortir. Aucune statistique (fréquences, retards, paires) ne permet de prédire le prochain tirage. L'application le montre elle-même avec un test du khi-deux et des backtests comparés à une stratégie purement aléatoire.
>
> Le seul levier réel est de **jouer des grilles peu populaires** : on ne gagne pas plus souvent, mais on partage moins le gain quand on gagne. C'est ce que mesure la colonne « Pénalité popularité ».

## Fonctionnalités

- Import de l'historique : archive ZIP, CSV ou Excel de la FDJ, ou récupération automatique depuis le site FDJ.
- Fréquences des numéros et des étoiles, test du khi-deux d'uniformité.
- Associations entre numéros et entre étoiles (tableaux et matrice de chaleur).
- Retards (nombre de tirages depuis la dernière apparition).
- Générateur de grilles (aléatoire, pondéré, équilibré) avec pénalité de popularité.
- Générateur multi-critères.
- Backtests chronologiques (simple et multi-périodes) avec score z pour savoir si un écart dépasse le hasard.

Les statistiques d'étoiles ne portent que sur les tirages depuis le 27/09/2016, date du passage à 12 étoiles (il y en avait 9, puis 11 avant).

## Lancer l'application en local

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Déployer sur Streamlit Community Cloud

1. Se connecter sur [share.streamlit.io](https://share.streamlit.io) avec son compte GitHub.
2. « Create app », choisir ce dépôt, la branche `main` et le fichier `app.py`.
3. Déployer.

## Organisation du code

| Fichier | Rôle |
|---|---|
| `app.py` | Interface Streamlit |
| `euromillions_core.py` | Calculs : lecture des fichiers, statistiques vectorisées (NumPy), générateur, tests statistiques, pénalité de popularité |
| `tests/test_core.py` | Tests automatiques (`pip install pytest` puis `pytest`) |

## Avertissement

Outil de curiosité statistique, sans valeur prédictive. Les jeux d'argent comportent des risques : endettement, dépendance. Pour de l'aide : Joueurs Info Service, 09 74 75 13 13 (appel non surtaxé).
