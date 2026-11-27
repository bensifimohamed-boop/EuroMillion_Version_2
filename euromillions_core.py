"""
euromillions_core.py
Fonctions de calcul corrigées et vectorisées (NumPy) pour l'application
Streamlit EuroMillions. À placer à côté de app.py puis :

    from euromillions_core import *

Elles remplacent les fonctions du même nom dans app.py
(mêmes colonnes en sortie, sauf mention contraire).
"""

import io
import zipfile
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st

NUMBER_COLUMNS = ["n1", "n2", "n3", "n4", "n5"]
STAR_COLUMNS = ["s1", "s2"]

# Date du passage à 12 étoiles (avant : 9 puis 11 étoiles).
# Toute statistique sur les étoiles doit partir de cette date.
DATE_12_ETOILES = pd.Timestamp("2016-09-27")


# ------------------------------------------------------------
# 1. Lecture des fichiers : correction de l'import Excel
# ------------------------------------------------------------

def read_table(content, filename):
    lower_name = filename.lower()

    # Un .xlsx est lui-même un ZIP (il commence par b"PK").
    # Il faut donc tester l'extension Excel AVANT le test ZIP,
    # sinon l'import Excel échoue toujours.
    if lower_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))

    if lower_name.endswith(".zip") or content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = [
                n for n in archive.namelist()
                if not n.endswith("/")
                and n.lower().endswith((".csv", ".xlsx", ".xls"))
            ]
            if not members:
                raise ValueError("Aucun fichier CSV ou Excel trouvé dans le ZIP.")
            errors = []
            for member in members:
                try:
                    return read_table(archive.read(member), member)
                except Exception as exc:
                    errors.append(f"{member}: {exc}")
            raise ValueError("Impossible de lire l'archive : " + " | ".join(errors))

    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        for separator in (None, ";", ",", "\t"):
            try:
                frame = pd.read_csv(
                    io.BytesIO(content), sep=separator,
                    encoding=encoding, engine="python",
                )
                if len(frame.columns) >= 5:
                    return frame
            except Exception:
                continue
    raise ValueError("Impossible de lire le fichier CSV.")


def valid_rows_mask(result):
    """Remplace la boucle ligne par ligne de normalize_draws."""
    nums = result[NUMBER_COLUMNS]
    stars = result[STAR_COLUMNS]
    return (
        nums.ge(1).all(axis=1) & nums.le(50).all(axis=1)
        & stars.ge(1).all(axis=1) & stars.le(12).all(axis=1)
        & nums.nunique(axis=1).eq(5)
        & stars.nunique(axis=1).eq(2)
    )


# ------------------------------------------------------------
# 2. Représentation « one-hot » : base de tous les calculs rapides
# ------------------------------------------------------------

def one_hot(draws, columns, size):
    """Matrice (tirages x size) : 1 si la valeur est sortie."""
    values = draws[columns].to_numpy(dtype=int) - 1
    matrix = np.zeros((len(draws), size), dtype=np.int32)
    rows = np.repeat(np.arange(len(draws)), len(columns))
    matrix[rows, values.ravel()] = 1
    return matrix


def star_draws(draws):
    """Tirages utilisables pour les statistiques d'étoiles (règle à 12)."""
    return draws[draws["date"] >= DATE_12_ETOILES]


# ------------------------------------------------------------
# 3. Statistiques (mises en cache : plus de recalcul à chaque clic)
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def pair_matrix(draws, columns, values):
    values = list(values)
    m = one_hot(draws, columns, len(values))
    matrix = m.T @ m                      # co-occurrences en une opération
    np.fill_diagonal(matrix, 0)
    return pd.DataFrame(matrix, index=values, columns=values)


@st.cache_data(show_spinner=False)
def pair_frequencies(draws, columns, total_values=50):
    nb_draws = len(draws)
    k = len(columns)
    m = one_hot(draws, columns, total_values)
    co = m.T @ m
    i, j = np.triu_indices(total_values, k=1)
    counts = co[i, j]
    keep = counts > 0
    probability = k * (k - 1) / (total_values * (total_values - 1))
    expected = nb_draws * probability
    result = pd.DataFrame({
        "Valeur 1": i[keep] + 1,
        "Valeur 2": j[keep] + 1,
        "Tirages ensemble": counts[keep],
    })
    result["Fréquence (%)"] = (result["Tirages ensemble"] / nb_draws * 100).round(2) if nb_draws else 0.0
    result["Attendu théorique"] = round(expected, 2)
    result["Écart observé-attendu"] = (result["Tirages ensemble"] - expected).round(2)
    return result.sort_values(
        ["Tirages ensemble", "Valeur 1", "Valeur 2"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def calculate_number_delays(draws, columns, possible_values):
    possible_values = list(possible_values)
    if draws.empty:
        return pd.DataFrame(columns=["Valeur", "Tirages depuis dernière apparition", "Statut"])
    m = one_hot(draws, columns, max(possible_values))[::-1]   # plus récent en premier
    seen = m.any(axis=0)
    delay = np.where(seen, m.argmax(axis=0), len(draws))
    frame = pd.DataFrame({"Valeur": possible_values,
                          "Tirages depuis dernière apparition": delay[np.array(possible_values) - 1]})
    frame["Statut"] = np.select(
        [~seen[np.array(possible_values) - 1], frame["Tirages depuis dernière apparition"].eq(0)],
        ["Jamais apparu dans l'historique", "Présent au dernier tirage"],
        "Absent des derniers tirages",
    )
    return frame.sort_values("Tirages depuis dernière apparition", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------
# 4. Test de correspondance vectorisé (≈ 1000x plus rapide)
# ------------------------------------------------------------

class HistoryIndex:
    """Pré-calcule les matrices une seule fois pour un jeu de tirages."""

    def __init__(self, draws):
        self.n = one_hot(draws, NUMBER_COLUMNS, 50)
        self.s = one_hot(draws, STAR_COLUMNS, 12)
        self.size = len(draws)

    def matches(self, numbers, stars):
        """Nombre de numéros et d'étoiles trouvés pour chaque tirage."""
        nm = self.n[:, np.asarray(numbers) - 1].sum(axis=1)
        sm = self.s[:, np.asarray(stars) - 1].sum(axis=1)
        return nm, sm

    def matches_many(self, grids_numbers, grids_stars):
        """Pour G grilles d'un coup : matrices (G x tirages)."""
        gn = np.zeros((len(grids_numbers), 50), dtype=np.int32)
        gs = np.zeros((len(grids_stars), 12), dtype=np.int32)
        for g, (nums, sts) in enumerate(zip(grids_numbers, grids_stars)):
            gn[g, np.asarray(nums) - 1] = 1
            gs[g, np.asarray(sts) - 1] = 1
        return gn @ self.n.T, gs @ self.s.T


# Rangs de gains EuroMillions (numéros, étoiles) -> rang.
PRIZE_TIERS = {
    (5, 2): 1, (5, 1): 2, (5, 0): 3, (4, 2): 4, (4, 1): 5, (3, 2): 6,
    (4, 0): 7, (2, 2): 8, (3, 1): 9, (3, 0): 10, (1, 2): 11, (2, 1): 12, (2, 0): 13,
}


def analyze_historical_performance(grid_row, historical_draws, index=None):
    """Même sortie que l'original, mais sans iterrows.
    Passer `index=HistoryIndex(draws)` pour réutiliser les matrices."""
    if historical_draws.empty:
        return {"Tirages analysés": 0}
    index = index or HistoryIndex(historical_draws)
    numbers = [int(grid_row[f"Numéro {i}"]) for i in range(1, 6)]
    stars = [int(grid_row[f"Étoile {i}"]) for i in range(1, 3)]
    nm, sm = index.matches(numbers, stars)
    total = len(nm)
    best = max(zip(nm.tolist(), sm.tolist()))
    results = {
        "Tirages analysés": total,
        "Meilleurs numéros trouvés": int(nm.max()),
        "Meilleures étoiles trouvées": int(sm.max()),
        "Meilleure combinaison": f"{best[0]} numéro(s) + {best[1]} étoile(s)",
        "Total numéros correspondants": int(nm.sum()),
        "Total étoiles correspondantes": int(sm.sum()),
        "Tirages avec au moins 1 numéro": int((nm >= 1).sum()),
        "Tirages avec au moins 1 étoile": int((sm >= 1).sum()),
    }
    for n in range(6):
        c = int((nm == n).sum())
        results[f"Tirages avec exactement {n} numéro(s)"] = c
        results[f"Pourcentage avec exactement {n} numéro(s)"] = c / total * 100
    for s in range(3):
        c = int((sm == s).sum())
        results[f"Tirages avec exactement {s} étoile(s)"] = c
        results[f"Pourcentage avec exactement {s} étoile(s)"] = c / total * 100
    for n in range(2, 6):
        results[f"Tirages avec {n} numéros"] = int((nm == n).sum())
        for s in range(1, 3):
            results[f"Tirages avec {n} numéros + {s} étoile(s)"] = int(((nm == n) & (sm == s)).sum())
    # Indicateur plus parlant que le « score cumulé » : nombre de tirages gagnants.
    results["Tirages gagnants (rang 1 à 13)"] = int(sum(
        ((nm == n) & (sm == s)).sum() for (n, s) in PRIZE_TIERS
    ))
    results["Score historique cumulé"] = int(nm.sum() + sm.sum())
    return results


# ------------------------------------------------------------
# 5. Générateur : comptages calculés une seule fois
# ------------------------------------------------------------

class GridGenerator:
    def __init__(self, draws, rng=None):
        self.rng = rng or np.random.default_rng()
        self.number_w = draws[NUMBER_COLUMNS].stack().value_counts().reindex(range(1, 51), fill_value=0).to_numpy() + 1.0
        sd = star_draws(draws)
        sd = sd if len(sd) else draws
        self.star_w = sd[STAR_COLUMNS].stack().value_counts().reindex(range(1, 13), fill_value=0).to_numpy() + 1.0

    def _pick(self, size, k, weights, weighted):
        p = weights / weights.sum() if weighted else None
        return np.sort(self.rng.choice(np.arange(1, size + 1), size=k, replace=False, p=p))

    def grid(self, weighted=True, balanced_parity=False, balanced_ranges=False):
        numbers = None
        for _ in range(500):
            cand = self._pick(50, 5, self.number_w, weighted)
            even = int((cand % 2 == 0).sum())
            low = int((cand <= 25).sum())
            if (not balanced_parity or even in (2, 3)) and (not balanced_ranges or low in (2, 3)):
                numbers = cand
                break
        if numbers is None:
            numbers = cand
        stars = self._pick(12, 2, self.star_w, weighted)
        return [int(x) for x in numbers] + [int(x) for x in stars]


def historical_keys(draws):
    """Clés (5 numéros triés + 2 étoiles triées), même format que les grilles.
    Corrige le bug du générateur multi-critères qui triait numéros ET étoiles
    ensemble, ce qui rendait l'exclusion inopérante."""
    nums = np.sort(draws[NUMBER_COLUMNS].to_numpy(dtype=int), axis=1)
    stars = np.sort(draws[STAR_COLUMNS].to_numpy(dtype=int), axis=1)
    return {tuple(n) + tuple(s) for n, s in zip(nums.tolist(), stars.tolist())}


# ------------------------------------------------------------
# 6. Test du hasard : khi-deux d'uniformité (sans SciPy)
# ------------------------------------------------------------

def _chi2_sf(x, dof):
    """p-valeur du khi-deux (approximation de Wilson-Hilferty, précise ici)."""
    from math import erf, sqrt
    z = ((x / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / sqrt(2 / (9 * dof))
    return 0.5 * (1 - erf(z / sqrt(2)))


def uniformity_test(draws, columns, size):
    counts = one_hot(draws, columns, size).sum(axis=0)
    expected = len(draws) * len(columns) / size
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    dof = size - 1
    return {"Khi-deux": round(chi2, 2), "Degrés de liberté": dof,
            "p-valeur": round(_chi2_sf(chi2, dof), 3)}


def backtest_significance(scores_strategy, scores_random):
    """Test apparié : l'écart moyen est-il plus grand que le bruit ?"""
    d = np.asarray(scores_strategy) - np.asarray(scores_random)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return {"Écart moyen": float(d.mean()) if len(d) else 0.0, "z": 0.0}
    z = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return {"Écart moyen": float(d.mean()), "z": float(z),
            "Significatif (|z| > 2)": bool(abs(z) > 2)}


# ------------------------------------------------------------
# 7. Ce qui améliore vraiment l'espérance : éviter les grilles populaires
# ------------------------------------------------------------

def popularity_penalty(numbers, stars, last_draw=None):
    """Pénalité (0 = grille peu jouée). Ne change PAS la probabilité de gagner,
    mais réduit le risque de partager le gain avec d'autres joueurs."""
    numbers = sorted(numbers)
    p = 0.0
    p += 1.0 * sum(n <= 31 for n in numbers) / 5          # dates d'anniversaire
    gaps = np.diff(numbers)
    p += 1.0 * (len(set(gaps.tolist())) == 1)             # suite arithmétique 1-2-3-4-5, 5-10-15...
    p += 0.5 * int((gaps == 1).sum() >= 2)                # plusieurs numéros consécutifs
    p += 0.3 * sum(n in (7, 3, 9, 11, 13, 17, 21) for n in numbers) / 5  # « numéros fétiches »
    p += 0.3 * sum(s <= 7 for s in stars) / 2
    if last_draw is not None:
        p += 1.0 * len(set(numbers) & set(last_draw)) / 5  # recopie du dernier tirage
    if all(n % 10 == numbers[0] % 10 for n in numbers):   # même colonne du bulletin
        p += 0.5
    return round(p, 3)
