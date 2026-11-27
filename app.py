import io
import re
import random
import unicodedata
import zipfile
from itertools import combinations
from urllib.parse import unquote, urljoin, urlparse

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

from euromillions_core import (
    DATE_12_ETOILES,
    GridGenerator,
    HistoryIndex,
    analyze_historical_performance,
    backtest_significance,
    calculate_number_delays,
    historical_keys,
    pair_frequencies,
    pair_matrix,
    popularity_penalty,
    read_table,
    star_draws,
    uniformity_test,
    valid_rows_mask,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EuroMillions - Analyse complète",
    page_icon="🎱",
    layout="wide",
)

FDJ_HISTORY_URL = (
    "https://www.fdj.fr/jeux-de-tirage/euromillions-my-million/historique"
)

COLUMNS = ["date", "n1", "n2", "n3", "n4", "n5", "s1", "s2"]
NUMBER_COLUMNS = ["n1", "n2", "n3", "n4", "n5"]
STAR_COLUMNS = ["s1", "s2"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/130.0 Safari/537.36"
    )
}


# ============================================================
# 2. OUTILS
# ============================================================

def norm(value):
    value = str(value).strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        c for c in value if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z0-9]", "", value)


def find_column(columns, candidates):
    normalized = {norm(c): c for c in columns}

    for candidate in candidates:
        if norm(candidate) in normalized:
            return normalized[norm(candidate)]

    return None


# ============================================================
# 3. LECTURE CSV, EXCEL ET ZIP
# ============================================================

# ============================================================
# 4. NORMALISATION DE L'HISTORIQUE
# ============================================================

def normalize_draws(frame):
    if frame is None or frame.empty:
        raise ValueError("Le fichier ne contient aucune ligne.")

    frame = frame.copy()
    frame.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in frame.columns
    ]

    date_col = find_column(
        frame.columns,
        ["date_de_tirage", "date tirage", "date", "jour_tirage"],
    )

    if date_col is None:
        raise ValueError("Colonne de date introuvable.")

    def locate(kind, number):
        if kind == "n":
            candidates = [
                f"boule_{number}",
                f"boule{number}",
                f"numero_{number}",
                f"numéro_{number}",
                f"numero{number}",
                f"boule gagnante {number}",
            ]
        else:
            candidates = [
                f"etoile_{number}",
                f"etoile{number}",
                f"étoile_{number}",
                f"numero_etoile_{number}",
                f"numero etoile {number}",
                f"étoile gagnante {number}",
            ]

        return find_column(frame.columns, candidates)

    mapping = {"date": date_col}

    for i in range(1, 6):
        mapping[f"n{i}"] = locate("n", i)

    for i in range(1, 3):
        mapping[f"s{i}"] = locate("s", i)

    missing = [
        key for key, value in mapping.items()
        if value is None
    ]

    if missing:
        raise ValueError(
            "Colonnes non reconnues : "
            + ", ".join(missing)
            + ". Colonnes disponibles : "
            + ", ".join(map(str, frame.columns))
        )

    result = pd.DataFrame()
    
        
    date_values = frame[mapping["date"]]

    if pd.api.types.is_numeric_dtype(date_values):
        result["date"] = pd.to_datetime(
            date_values.astype("Int64").astype(str),
            format="%Y%m%d",
            errors="coerce",
        )
    else:
        result["date"] = pd.to_datetime(
            date_values.astype(str).str.strip(),
            dayfirst=True,
            format="mixed",
            errors="coerce",
        )

    
    for col in NUMBER_COLUMNS + STAR_COLUMNS:
        result[col] = pd.to_numeric(
            frame[mapping[col]]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.strip(),
            errors="coerce",
        )

    result = result.dropna(subset=COLUMNS).copy()

    valid = valid_rows_mask(result)

    result = result.loc[valid, COLUMNS].copy()

    if result.empty:
        raise ValueError("Aucun tirage valide après vérification.")

    for col in NUMBER_COLUMNS + STAR_COLUMNS:
        result[col] = result[col].astype(int)

    return (
        result.drop_duplicates(subset=COLUMNS)
        .sort_values("date")
        .reset_index(drop=True)
    )


def parse_history_file(filename, content):
    return normalize_draws(read_table(content, filename)), filename


# ============================================================
# 5. ACTUALISATION FDJ
# ============================================================

@st.cache_data(ttl="12h", show_spinner=False)
def fetch_fdj_history():
    response = requests.get(
        FDJ_HISTORY_URL,
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    archive_urls = []

    for link in soup.find_all("a", href=True):
        href = urljoin(FDJ_HISTORY_URL, link["href"].strip())
        label = norm(link.get_text(" ", strip=True) + " " + href)

        is_archive = (
            ".zip" in href.lower()
            or "sto.api.fdj.fr" in href.lower()
            or ("telecharger" in label and "zip" in label)
        )

        if is_archive and href.startswith("https://"):
            if href not in archive_urls:
                archive_urls.append(href)

    if not archive_urls:
        raise ValueError(
            "Aucun lien d'archive FDJ détecté. "
            "Vous pouvez importer le ZIP manuellement."
        )

    all_draws = []
    diagnostics = []

    for url in archive_urls:
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=60,
            )
            response.raise_for_status()

            if not response.content.startswith(b"PK"):
                raise ValueError("Le contenu reçu n'est pas un ZIP valide.")

            filename = unquote(
                urlparse(url).path.rsplit("/", 1)[-1]
            ) or "fdj_archive.zip"

            if not filename.lower().endswith(".zip"):
                filename += ".zip"

            part, _ = parse_history_file(
                filename,
                response.content,
            )

            all_draws.append(part)
            diagnostics.append(
                f"{filename} : {len(part)} tirages"
            )

        except Exception as exc:
            diagnostics.append(
                f"{urlparse(url).path.rsplit('/', 1)[-1]} : {exc}"
            )

    if not all_draws:
        raise ValueError(
            "Aucune archive exploitable. "
            + " | ".join(diagnostics)
        )

    merged = pd.concat(all_draws, ignore_index=True)

    merged = (
        merged.drop_duplicates(subset=COLUMNS)
        .sort_values("date")
        .reset_index(drop=True)
    )

    return merged, diagnostics


# ============================================================
# 6. STATISTIQUES DE FRÉQUENCE
# ============================================================

def count_stats(draws, columns, values):
    counts = (
        draws[columns]
        .stack()
        .value_counts()
        .reindex(values, fill_value=0)
        .sort_index()
    )

    result = pd.DataFrame({
        "Valeur": counts.index.astype(int),
        "Occurrences": counts.values.astype(int),
    })

    total = int(result["Occurrences"].sum())

    result["Fréquence (%)"] = (
        (result["Occurrences"] / total * 100).round(2)
        if total
        else 0.0
    )

    return result


# ============================================================
# 7. ANALYSE DES PAIRES
# ============================================================


# ============================================================
# 8. GÉNÉRATION DE GRILLES
# ============================================================

def analyze_generated_grid(row):
    numbers = sorted([
        int(row["Numéro 1"]),
        int(row["Numéro 2"]),
        int(row["Numéro 3"]),
        int(row["Numéro 4"]),
        int(row["Numéro 5"]),
    ])

    even_count = sum(n % 2 == 0 for n in numbers)
    odd_count = 5 - even_count

    low_count = sum(n <= 25 for n in numbers)
    high_count = 5 - low_count

    gaps = [
        numbers[i + 1] - numbers[i]
        for i in range(len(numbers) - 1)
    ]

    consecutive_pairs = sum(gap == 1 for gap in gaps)

    return {
        "Somme des numéros": sum(numbers),
        "Pairs": even_count,
        "Impairs": odd_count,
        "Numéros 1–25": low_count,
        "Numéros 26–50": high_count,
        "Écart minimum": min(gaps),
        "Écart maximum": max(gaps),
        "Paires consécutives": consecutive_pairs,
    }

def calculate_grid_score(row):
    numbers = sorted([
        int(row["Numéro 1"]),
        int(row["Numéro 2"]),
        int(row["Numéro 3"]),
        int(row["Numéro 4"]),
        int(row["Numéro 5"]),
    ])

    # 1. Score de parité : maximum 30 points
    even_count = sum(n % 2 == 0 for n in numbers)

    parity_scores = {
        0: 0,
        1: 15,
        2: 30,
        3: 30,
        4: 15,
        5: 0,
    }

    parity_score = parity_scores[even_count]

    # 2. Répartition bas/haut : maximum 25 points
    low_count = sum(n <= 25 for n in numbers)

    range_scores = {
        0: 0,
        1: 12,
        2: 25,
        3: 25,
        4: 12,
        5: 0,
    }

    range_score = range_scores[low_count]

    # 3. Dispersion des numéros : maximum 25 points
    gaps = [
        numbers[i + 1] - numbers[i]
        for i in range(4)
    ]

    gap_range = max(gaps) - min(gaps)

    if gap_range >= 8:
        dispersion_score = 25
    elif gap_range >= 5:
        dispersion_score = 20
    elif gap_range >= 3:
        dispersion_score = 15
    elif gap_range >= 1:
        dispersion_score = 10
    else:
        dispersion_score = 5

    # 4. Diversité des zones : maximum 20 points
    zones = {
        min((n - 1) // 10, 4)
        for n in numbers
    }

    diversity_score = min(len(zones) * 5, 20)

    total_score = (
        parity_score
        + range_score
        + dispersion_score
        + diversity_score
    )

    return {
        "Score parité": parity_score,
        "Score répartition": range_score,
        "Score dispersion": dispersion_score,
        "Score diversité": diversity_score,
        "Score total /100": total_score,
    }

# ============================================================
# 9. INTERFACE
# ============================================================

st.title("🎱 EuroMillions — Tableau de bord")
st.caption(
    "Statistiques, associations historiques et génération de grilles."
)

if "draws" not in st.session_state:
    st.session_state.draws = None

if "source_name" not in st.session_state:
    st.session_state.source_name = ""

if "diagnostics" not in st.session_state:
    st.session_state.diagnostics = []

st.header("Charger l'historique")

uploaded_file = st.file_uploader(
    "Importer un ZIP, CSV ou fichier Excel FDJ",
    type=["zip", "csv", "xlsx", "xls"],
)

col_import, col_fdj = st.columns(2)

with col_import:
    if st.button("Charger le fichier importé", width="stretch"):
        if uploaded_file is None:
            st.warning("Sélectionnez d'abord un fichier.")
        else:
            try:
                loaded, filename = parse_history_file(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                )

                st.session_state.draws = loaded
                st.session_state.source_name = filename
                st.session_state.diagnostics = []

                st.success(f"{len(loaded)} tirages chargés.")

            except Exception as exc:
                st.error(f"Erreur de lecture : {exc}")

with col_fdj:
    if st.button("Actualiser depuis la FDJ", width="stretch"):
        with st.spinner("Récupération des archives FDJ..."):
            try:
                loaded, diagnostics = fetch_fdj_history()

                st.session_state.draws = loaded
                st.session_state.source_name = "Archives officielles FDJ"
                st.session_state.diagnostics = diagnostics

                st.success(f"{len(loaded)} tirages actualisés.")

            except Exception as exc:
                st.error(f"Actualisation impossible : {exc}")
                st.info(
                    "Vous pouvez télécharger l'archive FDJ "
                    "et l'importer manuellement."
                )

if st.session_state.diagnostics:
    with st.expander("Détail de la récupération FDJ"):
        for line in st.session_state.diagnostics:
            st.write("- " + line)

draws = st.session_state.draws

if draws is None or draws.empty:
    st.info(
        "Importez votre fichier historique ou actualisez depuis la FDJ."
    )
    st.stop()

st.caption(
    f"Source : {st.session_state.source_name} · "
    f"{len(draws)} tirages chargés"
)


# ============================================================
# 10. FILTRE DE PÉRIODE ET INDICATEURS
# ============================================================

st.header("Période d'analyse")

period = st.selectbox(
    "Choisir les tirages à analyser",
    [
        "Historique complet",
        "100 derniers tirages",
        "200 derniers tirages",
        "500 derniers tirages",
    ],
)

if period == "Historique complet":
    analysis_draws = draws.copy()
else:
    analysis_draws = draws.tail(int(period.split()[0])).copy()

# Les étoiles n'utilisent que les tirages à 12 étoiles (depuis le 27/09/2016),
# sinon les étoiles 10 à 12 paraissent artificiellement rares.
analysis_star_draws = star_draws(analysis_draws)
if analysis_star_draws.empty:
    analysis_star_draws = analysis_draws

metric1, metric2, metric3 = st.columns(3)

metric1.metric("Tirages analysés", len(analysis_draws))
metric2.metric(
    "Premier tirage",
    analysis_draws["date"].min().strftime("%d/%m/%Y"),
)
metric3.metric(
    "Dernier tirage",
    analysis_draws["date"].max().strftime("%d/%m/%Y"),
)


# ============================================================
# 11. TABLEAU DE BORD VISUEL
# ============================================================

st.header("Fréquences des numéros et des étoiles")

number_stats = count_stats(
    analysis_draws,
    NUMBER_COLUMNS,
    range(1, 51),
)

star_stats = count_stats(
    analysis_star_draws,
    STAR_COLUMNS,
    range(1, 13),
)

top_numbers = number_stats.sort_values(
    ["Occurrences", "Valeur"],
    ascending=[False, True],
).head(10)

top_stars = star_stats.sort_values(
    ["Occurrences", "Valeur"],
    ascending=[False, True],
)

chart1, chart2 = st.columns(2)

with chart1:
    st.subheader("10 numéros les plus fréquents")
    st.bar_chart(
        top_numbers.set_index("Valeur")[["Occurrences"]]
    )

with chart2:
    st.subheader("Fréquence des étoiles")
    st.bar_chart(
        top_stars.set_index("Valeur")[["Occurrences"]]
    )

st.subheader("Classement des 50 numéros")

ranking = number_stats.sort_values(
    ["Occurrences", "Valeur"],
    ascending=[False, True],
).reset_index(drop=True)

st.dataframe(ranking, width="stretch", hide_index=True)

st.subheader("Les tirages sont-ils vraiment aléatoires ?")

uniformity_numbers = uniformity_test(analysis_draws, NUMBER_COLUMNS, 50)
uniformity_stars = uniformity_test(analysis_star_draws, STAR_COLUMNS, 12)

u1, u2 = st.columns(2)
u1.metric(
    "Test du khi-deux sur les numéros (p-valeur)",
    f"{uniformity_numbers['p-valeur']:.3f}",
)
u2.metric(
    "Test du khi-deux sur les étoiles (p-valeur)",
    f"{uniformity_stars['p-valeur']:.3f}",
)
st.caption(
    "Une p-valeur supérieure à 0,05 signifie que les écarts de fréquence "
    "observés sont ceux qu'on attend du pur hasard : rien ne distingue "
    "l'historique d'un tirage parfaitement uniforme. Les étoiles sont "
    f"analysées depuis le {DATE_12_ETOILES:%d/%m/%Y} (passage à 12 étoiles)."
)


# ============================================================
# 12. NOUVEAUTÉ : ASSOCIATIONS ENTRE NUMÉROS
# ============================================================

st.divider()
st.header("🔗 Associations entre les numéros")

st.write(
    "Chaque paire correspond à deux numéros présents dans un même tirage. "
    "Le nombre indiqué correspond au nombre de tirages où la paire est apparue."
)

number_pairs = pair_frequencies(
    analysis_draws,
    NUMBER_COLUMNS,
)


st.subheader("Comparaison des paires : historique complet vs 100 derniers tirages")

if len(draws) >= 100:
    recent_draws = draws.tail(100).copy()

    all_pairs = pair_frequencies(
        draws,
        NUMBER_COLUMNS,
        total_values=50,
    )

    recent_pairs = pair_frequencies(
        recent_draws,
        NUMBER_COLUMNS,
        total_values=50,
    )

    comparison = all_pairs[
        ["Valeur 1", "Valeur 2", "Tirages ensemble", "Fréquence (%)"]
    ].merge(
        recent_pairs[
            ["Valeur 1", "Valeur 2", "Tirages ensemble", "Fréquence (%)"]
        ],
        on=["Valeur 1", "Valeur 2"],
        how="outer",
        suffixes=("_Historique", "_100_derniers"),
    ).fillna(0)

    comparison["Écart de fréquence (points)"] = (
        comparison["Fréquence (%)_100_derniers"]
        - comparison["Fréquence (%)_Historique"]
    ).round(2)

    comparison = comparison.sort_values(
        "Écart de fréquence (points)",
        ascending=False,
    ).reset_index(drop=True)

    st.write(
        "Les écarts positifs indiquent des paires plus fréquentes "
        "dans les 100 derniers tirages que dans l'historique complet. "
        "Les écarts négatifs indiquent l'inverse."
    )

    st.dataframe(
        comparison.head(20),
        width="stretch",
        hide_index=True,
    )

    comparison_csv = comparison.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "Télécharger la comparaison des paires en CSV",
        data=comparison_csv,
        file_name="comparaison_paires_100_derniers.csv",
        mime="text/csv",
    )

else:
    st.info(
        "Il faut au moins 100 tirages dans l'historique "
        "pour effectuer cette comparaison."
    )



if not number_pairs.empty:
    st.subheader("Les 20 paires de numéros les plus fréquentes")

    st.dataframe(
        number_pairs.head(20),
        width="stretch",
        hide_index=True,
    )

    pair_csv = number_pairs.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "Télécharger toutes les paires de numéros en CSV",
        data=pair_csv,
        file_name="associations_numeros.csv",
        mime="text/csv",
    )

    st.subheader("Matrice de chaleur des associations")

    st.write(
        "Chaque cellule indique combien de fois les deux numéros "
        "correspondants sont apparus ensemble. La diagonale est vide."
    )

    matrix = pair_matrix(
        analysis_draws,
        NUMBER_COLUMNS,
        range(1, 51),
    )

    st.dataframe(
        matrix.style.background_gradient(cmap="YlOrRd"),
        width="stretch",
        height=650,
    )

    matrix_csv = matrix.to_csv().encode("utf-8-sig")

    st.download_button(
        "Télécharger la matrice en CSV",
        data=matrix_csv,
        file_name="matrice_associations_numeros.csv",
        mime="text/csv",
    )
else:
    st.info("Aucune paire disponible pour cette période.")


st.header("Retard des numéros et des étoiles")

st.write(
    "Le retard correspond au nombre de tirages écoulés "
    "depuis la dernière apparition d'un numéro ou d'une étoile."
)

number_delays = calculate_number_delays(
    draws,
    NUMBER_COLUMNS,
    range(1, 51),
)

star_delays = calculate_number_delays(
    star_draws(draws) if not star_draws(draws).empty else draws,
    STAR_COLUMNS,
    range(1, 13),
)

tab_numbers, tab_stars = st.tabs(["Numéros", "Étoiles"])

with tab_numbers:
    st.subheader("Retard des 50 numéros")
    st.dataframe(
        number_delays,
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "Télécharger les retards des numéros en CSV",
        data=number_delays.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="retards_numeros.csv",
        mime="text/csv",
    )

with tab_stars:
    st.subheader("Retard des 12 étoiles")
    st.dataframe(
        star_delays,
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "Télécharger les retards des étoiles en CSV",
        data=star_delays.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="retards_etoiles.csv",
        mime="text/csv",
    )

# ============================================================
# 13. NOUVEAUTÉ : ASSOCIATIONS ENTRE ÉTOILES
# ============================================================

st.divider()
st.header("⭐ Associations entre étoiles")

star_pairs = pair_frequencies(
    analysis_star_draws,
    STAR_COLUMNS,
    total_values=12,
)

if not star_pairs.empty:
    st.write(
        "Il existe 66 paires possibles de deux étoiles distinctes "
        "parmi les 12 étoiles."
    )

    st.dataframe(
        star_pairs,
        width="stretch",
        hide_index=True,
    )

    star_pairs_csv = star_pairs.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "Télécharger les associations d'étoiles en CSV",
        data=star_pairs_csv,
        file_name="associations_etoiles.csv",
        mime="text/csv",
    )


# ============================================================
# 14. HISTORIQUE
# ============================================================

with st.expander("Consulter les 50 derniers tirages"):
    st.dataframe(
        draws.sort_values("date", ascending=False).head(50),
        width="stretch",
        hide_index=True,
    )


# ============================================================
# 15. EXPORT DES STATISTIQUES
# ============================================================

st.header("Exporter les statistiques")

number_csv = number_stats.rename(
    columns={"Valeur": "Numéro"}
).to_csv(index=False).encode("utf-8-sig")

st.download_button(
    "Télécharger les statistiques des numéros",
    data=number_csv,
    file_name="statistiques_euromillions.csv",
    mime="text/csv",
)

stars_csv = star_stats.rename(
    columns={"Valeur": "Étoile"}
).to_csv(index=False).encode("utf-8-sig")

st.download_button(
    "Télécharger les statistiques des étoiles",
    data=stars_csv,
    file_name="statistiques_etoiles_euromillions.csv",
    mime="text/csv",
)


# ============================================================
# 16. GÉNÉRATEUR DE GRILLES PERSONNALISÉ
# ============================================================

st.header("Générer des grilles personnalisées")

st.info(
    "Chaque combinaison valide a la même probabilité au prochain tirage. "
    "Les fréquences passées, les critères d’équilibre et l’exclusion de "
    "combinaisons déjà tirées modifient seulement la façon de générer les "
    "grilles ; ils ne permettent pas de mieux prédire le résultat."
)

number_of_grids = st.number_input(
    "Nombre de grilles à générer",
    min_value=1,
    max_value=100,
    value=5,
    step=1,
)

weighted_generation = st.checkbox(
    "Pondérer selon les fréquences historiques (sans valeur prédictive)",
    value=True,
)

balanced_parity = st.checkbox(
    "Favoriser un équilibre entre numéros pairs et impairs",
    value=False,
)

balanced_ranges = st.checkbox(
    "Favoriser un équilibre entre 1–25 et 26–50",
    value=False,
)

avoid_historical = st.checkbox(
    "Exclure les combinaisons déjà tirées (sans avantage de probabilité)",
    value=True,
)

if st.button("Générer les grilles", type="primary"):
    generated = []
    seen_grids = set()

    # Historique des combinaisons déjà tirées :
    # 5 numéros et 2 étoiles.
    historical_grids = set()

    if avoid_historical and not draws.empty:
        historical_grids = historical_keys(draws)

    generator = GridGenerator(analysis_draws)

    last_draw_numbers = draws[NUMBER_COLUMNS].iloc[-1].astype(int).tolist()
    max_attempts = int(number_of_grids) * 100
    attempts = 0

    while (
        len(generated) < int(number_of_grids)
        and attempts < max_attempts
    ):
        attempts += 1

        grid = generator.grid(
            weighted=weighted_generation,
            balanced_parity=balanced_parity,
            balanced_ranges=balanced_ranges,
        )

        numbers = tuple(sorted(grid[:5]))
        stars = tuple(sorted(grid[5:]))
        grid_key = numbers + stars

        # Ne pas produire deux fois la même grille.
        if grid_key in seen_grids:
            continue

        # Ne pas reproduire une combinaison historique,
        # si cette option est activée.
        if avoid_historical and grid_key in historical_grids:
            continue

        seen_grids.add(grid_key)

        generated.append({
            "Grille": len(generated) + 1,
            "Numéro 1": numbers[0],
            "Numéro 2": numbers[1],
            "Numéro 3": numbers[2],
            "Numéro 4": numbers[3],
            "Numéro 5": numbers[4],
            "Étoile 1": stars[0],
            "Étoile 2": stars[1],
            "Pénalité popularité": popularity_penalty(
                numbers, stars, last_draw=last_draw_numbers
            ),
        })

    if len(generated) < int(number_of_grids):
        st.warning(
            f"Seules {len(generated)} grilles distinctes ont pu être "
            f"générées sur les {int(number_of_grids)} demandées. "
            "Essayez de réduire le nombre de grilles ou de désactiver "
            "certaines contraintes."
        )

    st.session_state.generated_grids = pd.DataFrame(generated)


if "generated_grids" in st.session_state:
    st.subheader("Vos grilles")

    st.dataframe(
        st.session_state.generated_grids,
        width="stretch",
        hide_index=True,
    )

    st.subheader("Indicateurs descriptifs des grilles")
    st.subheader("Vue d'ensemble des indices descriptifs")

    score_values = st.session_state.generated_grids.apply(
        calculate_grid_score,
        axis=1,
        result_type="expand",
    )

    score_column = "Score total /100"
    scores = score_values[score_column]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Grilles générées",
        len(st.session_state.generated_grids),
    )

    col2.metric(
        "Indice descriptif le plus élevé",
        f"{scores.max():.0f}/100",
    )

    col3.metric(
        "Indice descriptif moyen",
        f"{scores.mean():.1f}/100",
    )

    col4.metric(
        "Indice descriptif le plus faible",
        f"{scores.min():.0f}/100",
    )

    st.subheader("Comparaison des indices descriptifs par grille")

    score_chart = pd.DataFrame({
        "Grille": [
            f"Grille {i}"
            for i in st.session_state.generated_grids["Grille"]
        ],
        "Indice descriptif": scores.tolist(),
    })

    score_chart = score_chart.sort_values(
        "Indice descriptif",
        ascending=False,
    )

    st.bar_chart(
        score_chart.set_index("Grille"),
        y="Indice descriptif",
    )
        
    grid_analysis = st.session_state.generated_grids.apply(
        analyze_generated_grid,
        axis=1,
        result_type="expand",
    )

    grid_scores = score_values

    # Combiner les indicateurs et les scores
    grid_analysis = pd.concat(
        [
            st.session_state.generated_grids[["Grille"]].reset_index(drop=True),
            grid_analysis.reset_index(drop=True),
            grid_scores.reset_index(drop=True),
        ],
        axis=1,
    )

    # Classer les grilles selon leur score total
    grid_analysis = grid_analysis.sort_values(
        "Score total /100",
        ascending=False,
    ).reset_index(drop=True)

    grid_analysis.insert(
        0,
        "Classement",
        range(1, len(grid_analysis) + 1),
    )

    st.write(
        "Le classement repose sur un indice descriptif des critères choisis. "
        "Il ne prédit pas les résultats futurs et ne mesure pas la probabilité "
        "de gain."
    )

    st.dataframe(
        grid_analysis.rename(
            columns={"Score total /100": "Indice descriptif /100"}
        ),
        width="stretch",
        hide_index=True,
    )

    # Préparer le fichier CSV avec les grilles et leurs analyses
    export_grids = st.session_state.generated_grids.copy()

    # Fusion sur le numéro de grille (et non collage ligne à ligne,
    # car grid_analysis a été trié par score).
    export_grids = export_grids.merge(
        grid_analysis.rename(
            columns={"Score total /100": "Indice descriptif /100"}
        ),
        on="Grille",
        how="left",
    )

    generated_csv = export_grids.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "Télécharger les grilles et leur analyse en CSV",
        data=generated_csv,
        file_name="grilles_euromillions_analyse.csv",
        mime="text/csv",
    )

# ========================================================
    # TEST HISTORIQUE DES GRILLES GÉNÉRÉES
    # ========================================================

    st.header("Test historique des grilles générées")

    st.write(
        "Chaque grille est comparée à tous les tirages historiques "
        "disponibles. Les résultats décrivent le passé et ne prédisent "
        "pas les prochains tirages."
    )

    if draws.empty:
        st.warning("Aucun tirage historique disponible.")

    elif st.session_state.generated_grids.empty:
        st.warning("Aucune grille à analyser.")

    else:
        all_draws_index = HistoryIndex(draws)
        historical_results = (
            st.session_state.generated_grids.apply(
                lambda row: analyze_historical_performance(
                    row, draws, all_draws_index
                ),
                axis=1,
                result_type="expand",
            )
        )

        historical_results.insert(
            0,
            "Grille",
            st.session_state.generated_grids["Grille"].values,
        )

        # ----------------------------------------------------
        # 1. INDICATEURS RÉCAPITULATIFS
        # ----------------------------------------------------

        st.subheader("Vue d'ensemble des résultats historiques")

        best_numbers = historical_results[
            "Meilleurs numéros trouvés"
        ].max()

        best_stars = historical_results[
            "Meilleures étoiles trouvées"
        ].max()

        total_draws = int(
            historical_results["Tirages analysés"].max()
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Tirages analysés",
            total_draws,
        )

        col2.metric(
            "Meilleur nombre de numéros",
            f"{best_numbers}/5",
        )

        col3.metric(
            "Meilleur nombre d'étoiles",
            f"{best_stars}/2",
        )

        col4.metric(
            "Meilleur score cumulé",
            int(
                historical_results[
                    "Score historique cumulé"
                ].max()
            ),
        )

        st.caption(
            "Le score cumulé additionne les correspondances "
            "de numéros et d'étoiles sur l'ensemble de l'historique."
        )

        # ----------------------------------------------------
        # 2. CLASSEMENT HISTORIQUE
        # ----------------------------------------------------

        st.subheader("Classement historique des grilles")

        historical_results = historical_results.sort_values(
            [
                "Score historique cumulé",
                "Meilleurs numéros trouvés",
                "Meilleures étoiles trouvées",
            ],
            ascending=False,
        ).reset_index(drop=True)

        historical_results.insert(
            0,
            "Classement historique",
            range(1, len(historical_results) + 1),
        )

        st.dataframe(
            historical_results,
            width="stretch",
            hide_index=True,
        )

        # ----------------------------------------------------
        # 3. GRAPHIQUE DES CORRESPONDANCES DE NUMÉROS
        # ----------------------------------------------------

        st.subheader(
            "Comparaison des correspondances de numéros"
        )

        number_columns = [
            "Tirages avec 2 numéros",
            "Tirages avec 3 numéros",
            "Tirages avec 4 numéros",
            "Tirages avec 5 numéros",
        ]

        number_chart = historical_results[
            ["Grille"] + number_columns
        ].copy()

        number_chart["Grille"] = number_chart[
            "Grille"
        ].apply(lambda value: f"Grille {value}")

        number_chart = number_chart.set_index("Grille")

        st.bar_chart(number_chart)

        # ----------------------------------------------------
        # 4. GRAPHIQUE DES SCORES CUMULÉS
        # ----------------------------------------------------

        st.subheader("Score historique cumulé par grille")

        cumulative_chart = historical_results[
            ["Grille", "Score historique cumulé"]
        ].copy()

        cumulative_chart["Grille"] = cumulative_chart[
            "Grille"
        ].apply(lambda value: f"Grille {value}")

        cumulative_chart = cumulative_chart.set_index("Grille")

        st.bar_chart(cumulative_chart)

        # ----------------------------------------------------
        # 5. EXPORT CSV
        # ----------------------------------------------------

        historical_csv = historical_results.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "Télécharger les résultats du test historique en CSV",
            data=historical_csv,
            file_name="euromillions_test_historique.csv",
            mime="text/csv",
        )

# ========================================================
# COMPARATEUR DES STRATÉGIES DE GÉNÉRATION
# ========================================================

st.header("Comparaison des stratégies de génération")

st.write(
    "Chaque stratégie est évaluée sur le même historique, "
    "avec les mêmes indicateurs statistiques."
)

strategy_grid_count = st.number_input(
    "Nombre de grilles par stratégie",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
    key="strategy_grid_count",
)

strategy_repetitions = st.number_input(
"Nombre de répétitions par stratégie",
min_value=1,
max_value=100,
value=10,
step=1,
key="strategy_repetitions",
)

if st.button(
    "Comparer les 4 stratégies",
    type="primary",
    key="compare_strategies",
):

    strategies = [
        {
            "Stratégie": "Aléatoire",
            "weighted": False,
            "balanced_parity": False,
            "balanced_ranges": False,
        },
        {
            "Stratégie": "Pondérée",
            "weighted": True,
            "balanced_parity": False,
            "balanced_ranges": False,
        },
        {
            "Stratégie": "Équilibrée",
            "weighted": False,
            "balanced_parity": True,
            "balanced_ranges": True,
        },
        {
            "Stratégie": "Combinée",
            "weighted": True,
            "balanced_parity": True,
            "balanced_ranges": True,
        },
    ]

    comparison_rows = []

    total_grids = (
        len(strategies)
        * int(strategy_grid_count)
        * int(strategy_repetitions)
    )

    generated_count = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    comparator_generator = GridGenerator(analysis_draws)
    comparator_index = HistoryIndex(draws)

    for strategy in strategies:

        repetition_scores = []
        repetition_number_matches = []
        repetition_star_matches = []
        repetition_best_numbers = []
        repetition_best_stars = []

        for repetition in range(
            int(strategy_repetitions)
        ):

            strategy_scores = []
            strategy_number_matches = 0
            strategy_star_matches = 0
            strategy_best_numbers = 0
            strategy_best_stars = 0

            for grid_number in range(
                int(strategy_grid_count)
            ):

                grid = comparator_generator.grid(
                    weighted=strategy["weighted"],
                    balanced_parity=strategy[
                        "balanced_parity"
                    ],
                    balanced_ranges=strategy[
                        "balanced_ranges"
                    ],
                )

                sorted_numbers = sorted(
                    int(number) for number in grid[:5]
                )

                sorted_stars = sorted(
                    int(star) for star in grid[5:7]
                )

                grid_row = {
                    f"Numéro {i + 1}": number
                    for i, number in enumerate(
                        sorted_numbers
                    )
                }

                grid_row.update({
                    f"Étoile {i + 1}": star
                    for i, star in enumerate(
                        sorted_stars
                    )
                })

                performance = (
                    analyze_historical_performance(
                        grid_row,
                        draws,
                        comparator_index,
                    )
                )

                strategy_scores.append(
                    performance[
                        "Score historique cumulé"
                    ]
                )

                strategy_number_matches += (
                    performance[
                        "Total numéros correspondants"
                    ]
                )

                strategy_star_matches += (
                    performance[
                        "Total étoiles correspondantes"
                    ]
                )

                strategy_best_numbers = max(
                    strategy_best_numbers,
                    performance[
                        "Meilleurs numéros trouvés"
                    ],
                )

                strategy_best_stars = max(
                    strategy_best_stars,
                    performance[
                        "Meilleures étoiles trouvées"
                    ],
                )

                generated_count += 1

                progress_bar.progress(
                    min(
                        generated_count / total_grids,
                        1.0,
                    )
                )

            # Résultats moyens de cette répétition
            if strategy_scores:
                repetition_scores.append(
                    sum(strategy_scores)
                    / len(strategy_scores)
                )

                repetition_number_matches.append(
                    strategy_number_matches
                    / len(strategy_scores)
                )

                repetition_star_matches.append(
                    strategy_star_matches
                    / len(strategy_scores)
                )

                repetition_best_numbers.append(
                    strategy_best_numbers
                )

                repetition_best_stars.append(
                    strategy_best_stars
                )

            status_text.text(
                f"Stratégie : {strategy['Stratégie']} | "
                f"Répétition {repetition + 1}/"
                f"{int(strategy_repetitions)}"
            )

        # Synthèse de toutes les répétitions
        comparison_rows.append({
            "Stratégie": strategy["Stratégie"],
            "Répétitions": int(strategy_repetitions),
            "Grilles par répétition": int(
                strategy_grid_count
            ),
            "Total de grilles": (
                int(strategy_grid_count)
                * int(strategy_repetitions)
            ),
            "Score historique moyen par grille": round(
                sum(repetition_scores)
                / len(repetition_scores),
                2,
            ) if repetition_scores else 0,
            "Score moyen minimal": round(
                min(repetition_scores),
                2,
            ) if repetition_scores else 0,
            "Score moyen maximal": round(
                max(repetition_scores),
                2,
            ) if repetition_scores else 0,
            "Écart-type des scores": round(
                pd.Series(repetition_scores).std(ddof=0),
                2,
            ) if repetition_scores else 0,
            "Moyenne des numéros correspondants": round(
                sum(repetition_number_matches)
                / len(repetition_number_matches),
                2,
            ) if repetition_number_matches else 0,
            "Moyenne des étoiles correspondantes": round(
                sum(repetition_star_matches)
                / len(repetition_star_matches),
                2,
            ) if repetition_star_matches else 0,
            "Meilleur nombre de numéros observé": max(
                repetition_best_numbers
            ) if repetition_best_numbers else 0,
            "Meilleur nombre d'étoiles observé": max(
                repetition_best_stars
            ) if repetition_best_stars else 0,
        })

    progress_bar.empty()
    status_text.success("Comparaison terminée !")

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    st.session_state.strategy_comparison = (
        comparison_df
    )

# Afficher les résultats conservés en mémoire
if "strategy_comparison" in st.session_state:

    comparison_df = (
        st.session_state.strategy_comparison
    )

    st.subheader("Résultats comparatifs")

    st.dataframe(
        comparison_df,
        width="stretch",
        hide_index=True,
    )

    st.subheader(
        "Score historique moyen par stratégie"
    )

    strategy_chart = comparison_df[
        [
            "Stratégie",
            "Score historique moyen par grille",
        ]
    ].set_index("Stratégie")

    st.bar_chart(strategy_chart)

    st.subheader(
        "Stabilité des résultats entre les répétitions"
    )

    stability_chart = comparison_df[
        [
            "Stratégie",
            "Score moyen minimal",
            "Score moyen maximal",
        ]
    ].set_index("Stratégie")

    st.bar_chart(stability_chart)

    comparison_csv = comparison_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "Télécharger la comparaison des stratégies en CSV",
        data=comparison_csv,
        file_name="euromillions_comparaison_strategies.csv",
        mime="text/csv",
        key="download_strategy_comparison",
    )

    st.caption(
        "Les résultats dépendent des grilles générées et de "
        "l'historique sélectionné par l'application. "
        "Ils ne prédisent pas les prochains tirages."
    )
# ============================================================
# 17. DEUXIÈME GÉNÉRATEUR : PROFIL MULTI-CRITÈRES
# ============================================================

st.header("Deuxième génération : profil multi-critères")
st.write(
    "Cette génération combine plusieurs statistiques historiques pour "
    "proposer des grilles conformes à un profil choisi. Elle ne prédit pas "
    "les prochains tirages et n'augmente pas les chances de gain."
)
st.info(
    "Les fréquences, associations et retards sont des descriptions du passé. "
    "Les résultats des backtests restent séparés : ils évaluent les quatre "
    "stratégies déjà présentes dans l'application et ne servent pas à choisir "
    "les prochains numéros ni à noter cette nouvelle génération."
)

with st.form("multi_criteria_generator_form"):
    multi_grid_count = st.number_input(
        "Nombre de grilles multi-critères",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
        key="multi_criteria_grid_count",
    )

    st.caption(
        "Choisis l'importance de chaque famille de critères. Un poids de 0 "
        "l'ignore ; les autres poids sont ramenés à une échelle commune."
    )

    weight_col1, weight_col2, weight_col3 = st.columns(3)
    with weight_col1:
        weight_frequency = st.slider(
            "Fréquences historiques",
            min_value=0,
            max_value=100,
            value=25,
            step=5,
            key="multi_weight_frequency",
        )
        weight_recent = st.slider(
            "Fréquences des 100 derniers tirages",
            min_value=0,
            max_value=100,
            value=15,
            step=5,
            key="multi_weight_recent",
        )
        weight_number_pairs = st.slider(
            "Associations entre numéros",
            min_value=0,
            max_value=100,
            value=15,
            step=5,
            key="multi_weight_number_pairs",
        )
    with weight_col2:
        weight_star_pairs = st.slider(
            "Associations entre étoiles",
            min_value=0,
            max_value=100,
            value=10,
            step=5,
            key="multi_weight_star_pairs",
        )
        weight_number_delay = st.slider(
            "Retard des numéros",
            min_value=0,
            max_value=100,
            value=10,
            step=5,
            key="multi_weight_number_delay",
        )
        weight_star_delay = st.slider(
            "Retard des étoiles",
            min_value=0,
            max_value=100,
            value=5,
            step=5,
            key="multi_weight_star_delay",
        )
    with weight_col3:
        weight_structure = st.slider(
            "Équilibre pairs/impairs et 1–25/26–50",
            min_value=0,
            max_value=100,
            value=10,
            step=5,
            key="multi_weight_structure",
        )
        multi_avoid_historical = st.checkbox(
            "Exclure les combinaisons déjà tirées (sans avantage probabiliste)",
            value=False,
            key="multi_avoid_historical",
        )
        generate_multi = st.form_submit_button(
            "Générer les grilles multi-critères",
            type="primary",
        )

if generate_multi:
    weights = {
        "Fréquences historiques": int(weight_frequency),
        "Fréquences récentes": int(weight_recent),
        "Associations numéros": int(weight_number_pairs),
        "Associations étoiles": int(weight_star_pairs),
        "Retard numéros": int(weight_number_delay),
        "Retard étoiles": int(weight_star_delay),
        "Équilibre": int(weight_structure),
    }
    total_weight = sum(weights.values())

    if total_weight == 0:
        st.error("Choisis au moins un poids supérieur à zéro.")
        st.session_state.pop("multi_criteria_grids", None)
    else:
        ordered_draws = draws.sort_values("date").reset_index(drop=True)
        recent_draws = ordered_draws.tail(100).copy()

        def normalized_map(values):
            numeric_values = {
                key: float(value)
                for key, value in values.items()
            }
            if not numeric_values:
                return {}
            low = min(numeric_values.values())
            high = max(numeric_values.values())
            if high == low:
                return {key: 0.5 for key in numeric_values}
            return {
                key: (value - low) / (high - low)
                for key, value in numeric_values.items()
            }

        def frequency_map(source_draws, columns, possible_values):
            counts = (
                source_draws[columns]
                .stack()
                .value_counts()
                .reindex(possible_values, fill_value=0)
                .to_dict()
            )
            return normalized_map(counts)

        def delay_map(source_draws, columns, possible_values):
            delay_frame = calculate_number_delays(
                source_draws,
                columns,
                possible_values,
            )
            delays = dict(zip(
                delay_frame["Valeur"].astype(int),
                delay_frame["Tirages depuis dernière apparition"].astype(float),
            ))
            return normalized_map(delays)

        def pair_map(source_draws, columns, possible_values):
            observed = pair_frequencies(
                source_draws,
                columns,
                total_values=len(possible_values),
            )
            values = {
                pair: 0.0
                for pair in combinations(possible_values, 2)
            }
            for v1, v2, together in zip(
                observed["Valeur 1"],
                observed["Valeur 2"],
                observed["Tirages ensemble"],
            ):
                values[(int(v1), int(v2))] = float(together)
            return normalized_map(values)

        all_number_frequency = frequency_map(
            ordered_draws, NUMBER_COLUMNS, range(1, 51)
        )
        star_era_draws = star_draws(ordered_draws)
        if star_era_draws.empty:
            star_era_draws = ordered_draws
        all_star_frequency = frequency_map(
            star_era_draws, STAR_COLUMNS, range(1, 13)
        )
        recent_number_frequency = frequency_map(
            recent_draws, NUMBER_COLUMNS, range(1, 51)
        )
        recent_star_frequency = frequency_map(
            recent_draws, STAR_COLUMNS, range(1, 13)
        )
        all_number_pairs = pair_map(
            ordered_draws, NUMBER_COLUMNS, range(1, 51)
        )
        recent_number_pairs = pair_map(
            recent_draws, NUMBER_COLUMNS, range(1, 51)
        )
        all_star_pairs = pair_map(
            star_era_draws, STAR_COLUMNS, range(1, 13)
        )
        recent_star_pairs = pair_map(
            recent_draws, STAR_COLUMNS, range(1, 13)
        )
        number_delays = delay_map(
            ordered_draws, NUMBER_COLUMNS, range(1, 51)
        )
        star_delays = delay_map(
            star_era_draws, STAR_COLUMNS, range(1, 13)
        )

        def mean_score(score_map, values):
            return sum(score_map.get(value, 0.5) for value in values) / len(values)

        # Clés au même format que les grilles candidates
        # (5 numéros triés puis 2 étoiles triées).
        excluded_keys = (
            historical_keys(ordered_draws)
            if multi_avoid_historical
            else set()
        )
        last_draw_numbers = ordered_draws[NUMBER_COLUMNS].iloc[-1].astype(int).tolist()

        random_pool = []
        seen_candidates = set(excluded_keys)
        pool_size = max(3000, int(multi_grid_count) * 200)
        pool_size = min(pool_size, 20000)

        for _ in range(pool_size):
            numbers = tuple(sorted(random.sample(range(1, 51), 5)))
            stars = tuple(sorted(random.sample(range(1, 13), 2)))
            grid_key = numbers + stars
            if grid_key in seen_candidates:
                continue
            seen_candidates.add(grid_key)

            number_pair_keys = list(combinations(numbers, 2))
            star_pair_key = tuple(stars)
            historical_frequency_score = (
                mean_score(all_number_frequency, numbers)
                + mean_score(all_star_frequency, stars)
            ) / 2
            recent_frequency_score = (
                mean_score(recent_number_frequency, numbers)
                + mean_score(recent_star_frequency, stars)
            ) / 2
            number_pair_score = (
                mean_score(all_number_pairs, number_pair_keys)
                + mean_score(recent_number_pairs, number_pair_keys)
            ) / 2
            star_pair_score = (
                all_star_pairs.get(star_pair_key, 0.5)
                + recent_star_pairs.get(star_pair_key, 0.5)
            ) / 2
            number_delay_score = mean_score(number_delays, numbers)
            star_delay_score = mean_score(star_delays, stars)

            even_count = sum(number % 2 == 0 for number in numbers)
            low_count = sum(number <= 25 for number in numbers)
            parity_balance = {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.0, 4: 0.5, 5: 0.0}[even_count]
            range_balance = {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.0, 4: 0.5, 5: 0.0}[low_count]
            balance_score = (parity_balance + range_balance) / 2

            feature_scores = {
                "Fréquences historiques": historical_frequency_score,
                "Fréquences récentes": recent_frequency_score,
                "Associations numéros": number_pair_score,
                "Associations étoiles": star_pair_score,
                "Retard numéros": number_delay_score,
                "Retard étoiles": star_delay_score,
                "Équilibre": balance_score,
            }
            conformity = sum(
                feature_scores[name] * weight
                for name, weight in weights.items()
            ) / total_weight

            random_pool.append({
                "numbers": numbers,
                "stars": stars,
                "Indice de conformité au profil (%)": round(conformity * 100, 1),
                "Fréquences historiques (%)": round(historical_frequency_score * 100, 1),
                "Fréquences récentes (%)": round(recent_frequency_score * 100, 1),
                "Associations numéros (%)": round(number_pair_score * 100, 1),
                "Associations étoiles (%)": round(star_pair_score * 100, 1),
                "Retard numéros (%)": round(number_delay_score * 100, 1),
                "Retard étoiles (%)": round(star_delay_score * 100, 1),
                "Équilibre (%)": round(balance_score * 100, 1),
                "Pénalité popularité": popularity_penalty(
                    numbers, stars, last_draw=last_draw_numbers
                ),
            })

        random_pool.sort(
            key=lambda candidate: candidate["Indice de conformité au profil (%)"],
            reverse=True,
        )
        selected = random_pool[:int(multi_grid_count)]
        multi_rows = []
        for index, candidate in enumerate(selected, start=1):
            row = {
                "Grille": index,
                **{
                    f"Numéro {number_index}": value
                    for number_index, value in enumerate(candidate["numbers"], start=1)
                },
                **{
                    f"Étoile {star_index}": value
                    for star_index, value in enumerate(candidate["stars"], start=1)
                },
            }
            row.update({
                key: value
                for key, value in candidate.items()
                if key not in ("numbers", "stars")
            })
            multi_rows.append(row)

        st.session_state["multi_criteria_grids"] = pd.DataFrame(multi_rows)
        st.session_state["multi_criteria_pool_mean"] = (
            sum(item["Indice de conformité au profil (%)"] for item in random_pool)
            / len(random_pool)
            if random_pool else 0.0
        )
        st.session_state["multi_criteria_weights"] = weights

if "multi_criteria_grids" in st.session_state:
    multi_grids_df = st.session_state["multi_criteria_grids"]
    st.subheader("Grilles multi-critères proposées")
    st.caption(
        "Les grilles sont classées selon leur conformité aux poids choisis. "
        "Cet indice n'est ni une probabilité de gain ni une prédiction."
    )
    st.dataframe(
        multi_grids_df,
        width="stretch",
        hide_index=True,
    )

    selected_mean = multi_grids_df[
        "Indice de conformité au profil (%)"
    ].mean()
    random_mean = st.session_state.get("multi_criteria_pool_mean", 0.0)
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric(
        "Indice moyen des grilles retenues",
        f"{selected_mean:.1f}%",
    )
    metric_b.metric(
        "Indice moyen du lot aléatoire candidat",
        f"{random_mean:.1f}%",
    )
    metric_c.metric(
        "Critères pondérés",
        sum(weight > 0 for weight in st.session_state.get("multi_criteria_weights", {}).values()),
    )
    st.caption(
        "La comparaison ci-dessus mesure uniquement l'adéquation au profil "
        "statistique choisi ; elle ne compare pas les chances de gagner."
    )

    multi_grids_csv = multi_grids_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Télécharger les grilles multi-critères en CSV",
        data=multi_grids_csv,
        file_name="euromillions_grilles_multi_criteres.csv",
        mime="text/csv",
        key="download_multi_criteria_grids",
    )

# ============================================================
# 18. BACKTEST HISTORIQUE CHRONOLOGIQUE
# ============================================================

st.header("Backtest historique : entraînement et test")

st.write(
    "Les 80 % de tirages les plus anciens servent à générer "
    "les grilles. Les 20 % les plus récents servent uniquement "
    "à évaluer les résultats."
)

backtest_grid_count = st.number_input(
    "Nombre de grilles par stratégie",
    min_value=5,
    max_value=100,
    value=20,
    step=5,
    key="backtest_grid_count",
)

if st.button(
    "Lancer le backtest historique",
    key="run_historical_backtest",
):

    # Utiliser l'historique complet, trié par date.
    backtest_draws = (
        draws.dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
        .copy()
    )

    split_index = int(len(backtest_draws) * 0.8)

    training_draws = backtest_draws.iloc[:split_index].copy()
    test_draws = backtest_draws.iloc[split_index:].copy()

    if len(training_draws) < 10 or test_draws.empty:

        st.error(
            "Historique insuffisant pour réaliser le backtest."
        )

    else:

        st.write(
            f"**Entraînement :** {len(training_draws)} tirages, "
            f"du {training_draws['date'].min():%d/%m/%Y} "
            f"au {training_draws['date'].max():%d/%m/%Y}."
        )

        st.write(
            f"**Test :** {len(test_draws)} tirages, "
            f"du {test_draws['date'].min():%d/%m/%Y} "
            f"au {test_draws['date'].max():%d/%m/%Y}."
        )

        backtest_strategies = [
            {
                "Stratégie": "Aléatoire",
                "weighted": False,
                "balanced_parity": False,
                "balanced_ranges": False,
            },
            {
                "Stratégie": "Pondérée",
                "weighted": True,
                "balanced_parity": False,
                "balanced_ranges": False,
            },
            {
                "Stratégie": "Équilibrée",
                "weighted": False,
                "balanced_parity": True,
                "balanced_ranges": True,
            },
            {
                "Stratégie": "Combinée",
                "weighted": True,
                "balanced_parity": True,
                "balanced_ranges": True,
            },
        ]

        backtest_results = []

        backtest_generator = GridGenerator(training_draws)
        test_index = HistoryIndex(test_draws)

        progress = st.progress(0)
        status = st.empty()

        total_iterations = (
            len(backtest_strategies)
            * int(backtest_grid_count)
        )
        completed = 0

        for strategy in backtest_strategies:

            grid_scores = []
            grid_number_matches = []
            grid_star_matches = []

            for _ in range(int(backtest_grid_count)):

                grid = backtest_generator.grid(
                    weighted=strategy["weighted"],
                    balanced_parity=strategy["balanced_parity"],
                    balanced_ranges=strategy["balanced_ranges"],
                )

                sorted_numbers = sorted(
                    int(number) for number in grid[:5]
                )

                sorted_stars = sorted(
                    int(star) for star in grid[5:7]
                )

                grid_row = {
                    f"Numéro {i + 1}": sorted_numbers[i]
                    for i in range(5)
                }

                grid_row.update({
                    f"Étoile {i + 1}": sorted_stars[i]
                    for i in range(2)
                })

                performance = analyze_historical_performance(
                    grid_row,
                    test_draws,
                    test_index,
                )

                
                # Normalisation par le nombre de tirages testés.
                number_matches_per_draw = (
                    performance["Total numéros correspondants"]
                    / len(test_draws)
                )

                star_matches_per_draw = (
                    performance["Total étoiles correspondantes"]
                    / len(test_draws)
                )

                score_per_draw = (
                    number_matches_per_draw
                    + star_matches_per_draw
                )

                grid_scores.append(score_per_draw)
                grid_number_matches.append(
                    number_matches_per_draw
                )
                grid_star_matches.append(
                    star_matches_per_draw
                )

                completed += 1
                progress.progress(
                    completed / total_iterations
                )
                status.text(
                    f"Calcul : {completed}/{total_iterations}"
                )

            backtest_results.append({
                "Stratégie": strategy["Stratégie"],
                "Grilles testées": len(grid_scores),
                "Score moyen par tirage": (
                    sum(grid_scores) / len(grid_scores)
                ),
                "Moyenne numéros correspondants par tirage": (
                    sum(grid_number_matches)
                    / len(grid_number_matches)
                ),
                "Moyenne étoiles correspondantes par tirage": (
                    sum(grid_star_matches)
                    / len(grid_star_matches)
                ),
                "Meilleur score observé par tirage": max(
                    grid_scores
                ),
            })

        backtest_df = pd.DataFrame(backtest_results)

        backtest_df = backtest_df.sort_values(
            "Score moyen par tirage",
            ascending=False,
        ).reset_index(drop=True)

        st.session_state["historical_backtest_results"] = (
            backtest_df
        )

        st.success("Backtest terminé.")

if "historical_backtest_results" in st.session_state:

    backtest_df = st.session_state[
        "historical_backtest_results"
    ]

    st.subheader("Résultats du backtest")

    st.dataframe(
        backtest_df,
        width="stretch",
        hide_index=True,
    )

    st.subheader("Score moyen par tirage de test")

    backtest_chart = backtest_df[
        ["Stratégie", "Score moyen par tirage"]
    ].set_index("Stratégie")

    backtest_chart.loc["Référence théorique aléatoire"] = 0.833333
    st.bar_chart(backtest_chart)

    backtest_csv = backtest_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "Télécharger les résultats du backtest en CSV",
        data=backtest_csv,
        file_name="euromillions_backtest_historique.csv",
        mime="text/csv",
        key="download_historical_backtest",
    )

    st.caption(
        "Le backtest mesure les correspondances sur des tirages "
        "passés non utilisés pour générer les grilles. Il ne "
        "permet pas de prédire les résultats futurs : les tirages "
        "EuroMillions restent aléatoires. Une grille valide obtient en moyenne "
        "environ 0,833 correspondance par tirage (numéros et étoiles). "
        "Les écarts observés peuvent donc venir du hasard."
    )

# ============================================================
# 19. BACKTEST MULTI-PÉRIODES
# ============================================================

st.header("Backtest multi-périodes : stabilité des stratégies")

st.write(
    "Ce test compare les quatre stratégies sur plusieurs périodes "
    "chronologiques. À chaque étape, seuls les tirages antérieurs "
    "à la période de test sont utilisés pour générer les grilles."
)

multi_backtest_grid_count = st.number_input(
    "Nombre de grilles par stratégie et par période",
    min_value=5,
    max_value=500,
    value=50,
    step=5,
    key="multi_backtest_grid_count",
)

multi_backtest_repetitions = st.number_input(
    "Nombre de répétitions indépendantes",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
    key="multi_backtest_repetitions",
)

if st.button(
    "Lancer le backtest multi-périodes",
    key="run_multi_backtest",
):

    multi_draws = (
        draws.dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
        .copy()
    )

   
    total_draws = len(multi_draws)

    # Pour commencer, nous exigeons suffisamment de données.
    if total_draws < 50:

        st.error(
            "Historique insuffisant : il faut au moins "
            "50 tirages pour ce backtest."
        )

    else:

        # Chaque période de test contient environ 10 % de l'historique.
        # Les bornes sont calculées sur les données chronologiques.
        split_points = [
            int(total_draws * 0.6),
            int(total_draws * 0.7),
            int(total_draws * 0.8),
            int(total_draws * 0.9),
            total_draws,
        ]

        # Éviter les frontières identiques sur les petits historiques.
        split_points = sorted(set(split_points))

        multi_strategies = [
            {
                "Stratégie": "Aléatoire",
                "weighted": False,
                "balanced_parity": False,
                "balanced_ranges": False,
            },
            {
                "Stratégie": "Pondérée",
                "weighted": True,
                "balanced_parity": False,
                "balanced_ranges": False,
            },
            {
                "Stratégie": "Équilibrée",
                "weighted": False,
                "balanced_parity": True,
                "balanced_ranges": True,
            },
            {
                "Stratégie": "Combinée",
                "weighted": True,
                "balanced_parity": True,
                "balanced_ranges": True,
            },
        ]

        multi_results = []

        progress_multi = st.progress(0)
        status_multi = st.empty()

        # Préparer les périodes chronologiques
        test_periods = []

        for i in range(len(split_points) - 1):

            train_end = split_points[i]
            test_end = split_points[i + 1]

            training = multi_draws.iloc[:train_end].copy()
            testing = multi_draws.iloc[train_end:test_end].copy()

            if len(training) < 10 or testing.empty:
                continue

            test_periods.append({
                "Période": len(test_periods) + 1,
                "training": training,
                "testing": testing,
                "generator": GridGenerator(training),
                "index": HistoryIndex(testing),
            })

        total_iterations = (
            len(test_periods)
            * len(multi_strategies)
            * int(multi_backtest_grid_count)
            * int(multi_backtest_repetitions)
        )

        completed_multi = 0

        for repetition in range(1, int(multi_backtest_repetitions) + 1):

            for period_info in test_periods:

                training = period_info["training"]
                testing = period_info["testing"]
                period_number = period_info["Période"]

                for strategy in multi_strategies:

                    grid_scores = []
                    number_matches_list = []
                    star_matches_list = []

                    # Répartition des correspondances pour les numéros (0 à 5).
                    number_distribution_counts = {
                        n: 0 for n in range(6)
                    }

                    # Répartition des correspondances pour les étoiles (0 à 2).
                    star_distribution_counts = {
                        s: 0 for s in range(3)
                    }
                    
                    

                    for _ in range(int(multi_backtest_grid_count)):

                        grid = period_info["generator"].grid(
                            weighted=strategy["weighted"],
                            balanced_parity=strategy["balanced_parity"],
                            balanced_ranges=strategy["balanced_ranges"],
                        )

                        numbers = sorted(
                            int(number) for number in grid[:5]
                        )

                        stars = sorted(
                            int(star) for star in grid[5:7]
                        )

                        grid_row = {
                            f"Numéro {i + 1}": numbers[i]
                            for i in range(5)
                        }

                        grid_row.update({
                            f"Étoile {i + 1}": stars[i]
                            for i in range(2)
                        })

                        performance = analyze_historical_performance(
                            grid_row,
                            testing,
                            period_info["index"],
                        )

                        # Cumul des fréquences de correspondance des numéros.
                        for n in range(6):
                            number_distribution_counts[n] += performance[
                                f"Tirages avec exactement {n} numéro(s)"
                            ]

                        # Cumul des fréquences de correspondance des étoiles.
                        for s in range(3):
                            star_distribution_counts[s] += performance[
                                f"Tirages avec exactement {s} étoile(s)"
                            ]

                        number_avg = (
                            performance["Total numéros correspondants"]
                            / len(testing)
                        )

                        star_avg = (
                            performance["Total étoiles correspondantes"]
                            / len(testing)
                        )

                        score_avg = number_avg + star_avg

                        grid_scores.append(score_avg)
                        number_matches_list.append(number_avg)
                        star_matches_list.append(star_avg)

                        completed_multi += 1

                        progress_multi.progress(
                            completed_multi / total_iterations
                        )

                        status_multi.text(
                            f"Répétition {repetition}/"
                            f"{int(multi_backtest_repetitions)} — "
                            f"Période {period_number} — "
                            f"{strategy['Stratégie']} : "
                            f"{completed_multi}/{total_iterations}"
                        )

                    multi_results.append({
                        "Répétition": repetition,
                        "Période": period_number,
                        "Stratégie": strategy["Stratégie"],
                        "Début entraînement": (
                            training["date"].min().strftime("%d/%m/%Y")
                        ),
                        "Fin entraînement": (
                            training["date"].max().strftime("%d/%m/%Y")
                        ),
                        "Début test": (
                            testing["date"].min().strftime("%d/%m/%Y")
                        ),
                        "Fin test": (
                            testing["date"].max().strftime("%d/%m/%Y")
                        ),
                        "Tirages testés": len(testing),
                        "Grilles testées": len(grid_scores),
                        "Score moyen par tirage": (
                            sum(grid_scores) / len(grid_scores)
                        ),
                        "Moyenne numéros correspondants": (
                            sum(number_matches_list)
                            / len(number_matches_list)
                        ),
                        "Moyenne étoiles correspondantes": (
                            sum(star_matches_list)
                            / len(star_matches_list)
                        ),
                        **{
                            f"Tirages avec exactement {n} numéro(s)": count
                            for n, count in number_distribution_counts.items()
                        },
                        **{
                            f"Tirages avec exactement {s} étoile(s)": count
                            for s, count in star_distribution_counts.items()
                        },
                    })
        multi_results_df = pd.DataFrame(multi_results)

        if not multi_results_df.empty:

            st.session_state["multi_backtest_results"] = (
                multi_results_df
            )

            st.success(
                "Backtest multi-périodes terminé."
            )

        else:
            st.error(
                "Aucune période de test exploitable n'a été créée."
            )


if "multi_backtest_results" in st.session_state:

    multi_results_df = st.session_state["multi_backtest_results"]
    
 # Regrouper les répétitions pour obtenir une moyenne
    # par période et par stratégie.
    multi_summary_df = (
        multi_results_df.groupby(
            ["Période", "Stratégie"],
            as_index=False,
        )
        .agg(
            **{
                "Score moyen par tirage": (
                    "Score moyen par tirage",
                    "mean",
                ),
                "Moyenne numéros correspondants": (
                    "Moyenne numéros correspondants",
                    "mean",
                ),
                "Moyenne étoiles correspondantes": (
                    "Moyenne étoiles correspondantes",
                    "mean",
                ),
                "Écart-type entre répétitions": (
                    "Score moyen par tirage",
                    "std",
                ),
                "Nombre de répétitions": (
                    "Répétition",
                    "nunique",
                ),
            }
        )
    )
    # Répartition détaillée des correspondances
    st.subheader("Répartition des numéros et des étoiles correspondants")

    distribution_rows = []

    for (period, strategy_name), group in multi_results_df.groupby(
        ["Période", "Stratégie"]
    ):
        total_number_observations = sum(
            group[f"Tirages avec exactement {n} numéro(s)"].sum()
            for n in range(6)
        )

        total_star_observations = sum(
            group[f"Tirages avec exactement {s} étoile(s)"].sum()
            for s in range(3)
        )

        row = {
            "Période": period,
            "Stratégie": strategy_name,
        }

        for n in range(6):
            count = int(
                group[f"Tirages avec exactement {n} numéro(s)"].sum()
            )

            percentage = (
                count / total_number_observations * 100
                if total_number_observations > 0 else 0
            )

            row[f"{n} numéro(s)"] = count
            row[f"{n} numéro(s) (%)"] = round(percentage, 2)

        for s in range(3):
            count = int(
                group[f"Tirages avec exactement {s} étoile(s)"].sum()
            )

            percentage = (
                count / total_star_observations * 100
                if total_star_observations > 0 else 0
            )

            row[f"{s} étoile(s)"] = count
            row[f"{s} étoile(s) (%)"] = round(percentage, 2)

        distribution_rows.append(row)

    distribution_df = pd.DataFrame(distribution_rows)

    st.markdown("**A. Répartition des numéros correspondants**")

    number_columns = (
        ["Période", "Stratégie"]
        + [
            column
            for n in range(6)
            for column in (f"{n} numéro(s)", f"{n} numéro(s) (%)")
        ]
    )

    st.dataframe(
        distribution_df[number_columns],
        width="stretch",
        hide_index=True,
    )

    st.markdown("**B. Répartition des étoiles correspondantes**")

    star_columns = (
        ["Période", "Stratégie"]
        + [
            column
            for s in range(3)
            for column in (f"{s} étoile(s)", f"{s} étoile(s) (%)")
        ]
    )

    st.dataframe(
        distribution_df[star_columns],
        width="stretch",
        hide_index=True,
    )

    st.subheader("Comparaison appariée avec la stratégie aléatoire")

    paired_scores = multi_results_df.pivot_table(
        index=["Période", "Répétition"],
        columns="Stratégie",
        values="Score moyen par tirage",
        aggfunc="mean",
    )

    if "Aléatoire" in paired_scores.columns:

        paired_comparisons = []

        for strategy_name in paired_scores.columns:

            if strategy_name == "Aléatoire":
                continue

            differences = (
                paired_scores[strategy_name]
                - paired_scores["Aléatoire"]
            ).dropna()

            if not differences.empty:
                repetitions_above_random = int((differences > 0).sum())
                significance = backtest_significance(
                    paired_scores.loc[differences.index, strategy_name],
                    paired_scores.loc[differences.index, "Aléatoire"],
                )

                paired_comparisons.append({
                    "Stratégie": strategy_name,
                    "Écart moyen vs aléatoire": differences.mean(),
                    "Écart médian vs aléatoire": differences.median(),
                    "Répétitions avec un score supérieur": repetitions_above_random,
                    "Répétitions comparées": len(differences),
                    "Écart minimal": differences.min(),
                    "Écart maximal": differences.max(),
                    "Score z": round(significance.get("z", 0.0), 2),
                    "Écart significatif (|z| > 2)": significance.get(
                        "Significatif (|z| > 2)", False
                    ),
                })
        paired_comparison_df = pd.DataFrame(
            paired_comparisons
        )

        paired_comparison_df = paired_comparison_df.sort_values(
            "Écart moyen vs aléatoire",
            ascending=False,
        )

        st.dataframe(
            paired_comparison_df,
            width="stretch",
            hide_index=True,
        )

        st.subheader(
            "Part des répétitions où la stratégie dépasse l’aléatoire, par période"
        )

        period_win_rows = []

        for period in paired_scores.index.get_level_values(
            "Période"
        ).unique():

            period_data = paired_scores.xs(
                period,
                level="Période",
            )

            for strategy_name in paired_scores.columns:

                if strategy_name == "Aléatoire":
                    continue

                differences = (
                    period_data[strategy_name]
                    - period_data["Aléatoire"]
                ).dropna()

                if not differences.empty:

                    wins = int((differences > 0).sum())
                    ties = int((differences == 0).sum())
                    total = len(differences)

                    success_share = wins / total * 100

                    period_win_rows.append({
                        "Période": period,
                        "Stratégie": strategy_name,
                        "Répétitions supérieures": wins,
                        "Égalités": ties,
                        "Comparaisons": total,
                        "Part au-dessus du hasard (%)": success_share,
                    })
        period_win_df = pd.DataFrame(period_win_rows)

        if not period_win_df.empty:

            st.dataframe(
                period_win_df,
                width="stretch",
                hide_index=True,
            )

        else:
            st.info(
                "Aucune comparaison disponible pour calculer "
                "le taux de réussite."
            )
        st.subheader("Évolution des écarts par période")

        period_scores = multi_summary_df.pivot(
            index="Période",
            columns="Stratégie",
            values="Score moyen par tirage",
        )

        period_differences = (
            period_scores.drop(columns=["Aléatoire"])
            .subtract(
                period_scores["Aléatoire"],
                axis=0,
            )
        )

        st.write(
            "Écart moyen de chaque stratégie par rapport "
            "à la stratégie Aléatoire :"
        )

        st.dataframe(
            period_differences,
            width="stretch",
        )

        st.line_chart(
            period_differences,
            width="stretch",
        )

    else:
        st.warning(
            "La stratégie Aléatoire est absente des résultats."
        )
    st.subheader("Résultats détaillés par période")
    st.dataframe(
        multi_results_df,
        width="stretch",
        hide_index=True,
    )

    st.subheader("Score moyen par stratégie et par période")

    score_pivot = multi_summary_df.pivot(
        index="Période",
        columns="Stratégie",
        values="Score moyen par tirage",
    )

    score_pivot = score_pivot.apply(
        pd.to_numeric,
        errors="coerce",
    )

    score_pivot["Référence théorique aléatoire"] = 0.833333

    st.write("Valeurs du score moyen par stratégie et par période :")
    st.dataframe(
        score_pivot,
        width="stretch",
    )

    if not score_pivot.empty and score_pivot.notna().any().any():
        st.line_chart(
            score_pivot,
            width="stretch",
        )
    else:
        st.warning(
            "Aucune donnée numérique disponible pour afficher le graphique."
        )

    st.subheader("Stabilité des stratégies entre répétitions")

    stability_pivot = multi_summary_df.pivot(
        index="Période",
        columns="Stratégie",
        values="Écart-type entre répétitions",
    )

    stability_pivot = stability_pivot.apply(
        pd.to_numeric,
        errors="coerce",
    )

    st.write("Écart-type des scores entre répétitions :")

    st.dataframe(
        stability_pivot,
        width="stretch",
    )

    if not stability_pivot.empty and stability_pivot.notna().any().any():
        st.line_chart(
            stability_pivot,
            width="stretch",
        )
    else:
        st.info(
            "Pas assez de répétitions pour calculer la stabilité."
        )
    
    st.subheader("Classement global des stratégies")

    strategy_summary = (
        multi_summary_df.groupby("Stratégie")
        .agg(
            Score_moyen=(
                "Score moyen par tirage",
                "mean",
            ),
            Ecart_type_repetitions=(
                "Écart-type entre répétitions",
                "mean",
            ),
            Score_min=(
                "Score moyen par tirage",
                "min",
            ),
            Score_max=(
                "Score moyen par tirage",
                "max",
            ),
            Nombre_de_periodes=(
                "Période",
                "nunique",
            ),
        )
        .reset_index()
    )

    strategy_summary["Ecart_vs_reference"] = (
        strategy_summary["Score_moyen"] - 0.833333
    )

    strategy_summary = strategy_summary.sort_values(
        "Score_moyen",
        ascending=False,
    )

    strategy_summary = strategy_summary.rename(
        columns={
            "Score_moyen": "Score moyen global",
            "Ecart_type_repetitions": "Écart-type entre répétitions",
            "Ecart_vs_reference": "Écart vs référence théorique",
            "Score_min": "Score minimal observé",
            "Score_max": "Score maximal observé",
            "Nombre_de_periodes": "Nombre de périodes",
        }
    )

    st.dataframe(
        strategy_summary,
        width="stretch",
        hide_index=True,
    )
    
    st.subheader("Comparaison avec la stratégie aléatoire")

    random_reference = (
    multi_summary_df[
        multi_summary_df["Stratégie"] == "Aléatoire"
    ][["Période", "Score moyen par tirage"]]
        .rename(
            columns={
                "Score moyen par tirage": "Score de référence"
            }
        )
    )

    strategy_comparison = multi_summary_df.merge(
        random_reference,
        on="Période",
        how="left",
    )

    strategy_comparison["Écart vs aléatoire"] = (
        strategy_comparison["Score moyen par tirage"]
        - strategy_comparison["Score de référence"]
    )

    strategy_comparison = strategy_comparison[
        strategy_comparison["Stratégie"] != "Aléatoire"
    ]

    strategy_comparison = strategy_comparison[
        [
            "Période",
            "Stratégie",
            "Score moyen par tirage",
            "Score de référence",
            "Écart vs aléatoire",
        ]
    ]

    st.dataframe(
        strategy_comparison,
        width="stretch",
        hide_index=True,
    )

    st.subheader("Comparaison des scores moyens globaux")

    summary_chart = strategy_summary[
        ["Stratégie", "Score moyen global"]
    ].set_index("Stratégie")

    st.bar_chart(
        summary_chart,
        width="stretch",
    )

    multi_csv = multi_results_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "Télécharger le backtest multi-périodes en CSV",
        data=multi_csv,
        file_name="euromillions_backtest_multi_periodes.csv",
        mime="text/csv",
        key="download_multi_backtest",
    )

    st.caption(
        "Ce test compare des résultats historiques sur plusieurs périodes. "
        "Il ne permet pas de prédire les prochains tirages. "
        "Les stratégies sont évaluées sur les mêmes périodes de test. "
        "Une grille valide obtient en moyenne environ 0,833 correspondance "
        "par tirage ; les écarts observés peuvent venir du hasard."
    )

    
    
# ============================================================
# 20. RAPPEL
# ============================================================

st.divider()

st.caption(
    "Les associations affichées décrivent uniquement les tirages passés. "
    "Elles ne prédisent pas les résultats futurs : chaque tirage est "
    "indépendant et toutes les combinaisons valides ont la même probabilité."
)

