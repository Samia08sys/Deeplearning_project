"""
Part I — Data Preparation
Dataset: Star Classification Dataset (Kaggle: deepu1109/star-dataset)
  → Téléchargement automatique via kagglehub si disponible
  → Fallback : placez manuellement 'Stars.csv' dans outputs/part1/

Features : Temperature, Luminosity(L/Lo), Radius(R/Ro),
           Absolute Magnitude(Mv), Star Color, Spectral Class
Target   : Star Type  (0=Brown Dwarf … 5=Hypergiant)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── expected column names (Kaggle CSV) ────────────────────────────────────────
# The real Kaggle CSV has these exact columns:
KAGGLE_COLS = {
    "Temperature (K)": "Temperature",
    "Luminosity(L/Lo)": "Luminosity",
    "Radius(R/Ro)": "Radius",
    "Absolute magnitude(Mv)": "Absolute Magnitude",
    "Star color": "Star Color",
    "Spectral Class": "Spectral Class",
    "Star type": "Star Type",
}

OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs", "part1")
DATA_PATH = os.path.join(OUT_DIR, "Stars.csv")

# Star type labels (Kaggle encoding)
CLASS_LABELS = {
    0: "Brown Dwarf",
    1: "Red Dwarf",
    2: "White Dwarf",
    3: "Main Sequence",
    4: "Supergiant",
    5: "Hypergiant",
}


# ─────────────────────────────────────────────────────────────────────────────
def _try_kaggle_download() -> str | None:
    """Download the real dataset via kagglehub.
    Returns the CSV path on success, None otherwise.
    """
    try:
        import kagglehub
        print("[Data] Téléchargement via kagglehub (deepu1109/star-dataset)…")
        path = kagglehub.dataset_download("deepu1109/star-dataset")
        print(f"[Data] Dataset Kaggle téléchargé dans : {path}")
        # Find the CSV inside downloaded folder
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith(".csv"):
                    return os.path.join(root, f)
    except ImportError:
        print("[Data] kagglehub non installé  (pip install kagglehub)")
    except Exception as e:
        print(f"[Data] kagglehub échoué : {e}")
        print("[Data] → Vérifiez que votre clé Kaggle API est configurée :")
        print("         https://www.kaggle.com/settings  → Create New Token")
        print("         Placez kaggle.json dans ~/.kaggle/  ou définissez")
        print("         KAGGLE_USERNAME / KAGGLE_KEY comme variables d'env.")
    return None


def _manual_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  Téléchargement manuel du dataset Kaggle                        ║
╠══════════════════════════════════════════════════════════════════╣
║  1. Allez sur : https://www.kaggle.com/datasets/deepu1109/      ║
║                  star-dataset                                   ║
║  2. Cliquez sur « Download Dataset »  (zip ≈ 3 KB)             ║
║  3. Extrayez le fichier ZIP                                     ║
║  4. Copiez  Stars.csv  dans :                                   ║
║       outputs/part1/Stars.csv                                   ║
║  5. Relancez :  python part1_mlp/main.py                        ║
╚══════════════════════════════════════════════════════════════════╝
""")


def _generate_synthetic(n_samples: int = 240) -> pd.DataFrame:
    """Fallback : génère des données synthétiques réalistes."""
    rng = np.random.RandomState(SEED)
    params = [
        (2500,  0.0001, 0.1,  16.0, "Red",          "M"),
        (3500,  0.01,   0.4,  11.0, "Red",          "M"),
        (9000,  0.001,  0.06,  9.0, "White",        "A"),
        (7000,  1.0,    1.0,   5.0, "Yellow-White", "F"),
        (20000, 50000., 14.0, -3.0, "Blue",         "O"),
        (40000, 1e6,    100., -8.5, "Blue-white",   "O"),
    ]
    n_per = n_samples // 6
    rows  = []
    for stype, (T_m, L_m, R_m, M_m, col, sp) in enumerate(params):
        for _ in range(n_per):
            rows.append({
                "Temperature":        abs(rng.normal(T_m, T_m * 0.1)),
                "Luminosity":         abs(rng.normal(L_m, abs(L_m) * 0.1)),
                "Radius":             abs(rng.normal(R_m, R_m * 0.1)),
                "Absolute Magnitude": rng.normal(M_m, 0.5),
                "Star Color":         col,
                "Spectral Class":     sp,
                "Star Type":          stype,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
def _load_raw_df() -> pd.DataFrame:
    """Charge le CSV réel (Kaggle ou copie manuelle) ou génère des données."""
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. CSV déjà présent localement ?
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print(f"[Data] ✓ CSV réel trouvé : {DATA_PATH}  shape={df.shape}")
        return df

    # 2. Tentative de téléchargement automatique
    csv_path = _try_kaggle_download()
    if csv_path and os.path.exists(csv_path):
        import shutil
        shutil.copy(csv_path, DATA_PATH)
        df = pd.read_csv(DATA_PATH)
        print(f"[Data] ✓ Dataset Kaggle chargé  shape={df.shape}")
        return df

    # 3. Fallback synthétique
    _manual_instructions()
    print("[Data] ⚠ Utilisation de données SYNTHÉTIQUES en attendant le CSV réel.")
    df = _generate_synthetic(240)
    df.to_csv(DATA_PATH, index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes Kaggle vers les noms internes standards."""
    df = df.rename(columns=KAGGLE_COLS)
    # Garde uniquement les colonnes attendues
    expected = list(KAGGLE_COLS.values())
    present  = [c for c in expected if c in df.columns]
    missing  = [c for c in expected if c not in df.columns]
    if missing:
        print(f"[Data] ⚠ Colonnes manquantes (ignorées) : {missing}")
    return df[present]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage : suppression NaN, doublons, valeurs aberrantes."""
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna()
    # Supprimer les luminosités / rayons négatifs (erreurs de saisie)
    for col in ["Luminosity", "Radius", "Temperature"]:
        if col in df.columns:
            df = df[df[col] > 0]
    after = len(df)
    print(f"[Data] Nettoyage : {before} → {after} lignes  "
          f"({before - after} supprimées)")
    return df.reset_index(drop=True)


def _show_stats(df: pd.DataFrame):
    """Affiche les statistiques descriptives du dataset."""
    print("\n[Data] ── Aperçu des données ──")
    print(df.head())
    print("\n[Data] ── Statistiques numériques ──")
    print(df.describe())
    print("\n[Data] ── Distribution des classes ──")
    counts = df["Star Type"].value_counts().sort_index()
    for idx, cnt in counts.items():
        label = CLASS_LABELS.get(int(idx), str(idx))
        bar   = "█" * (cnt * 20 // counts.max())
        print(f"  [{int(idx)}] {label:15s}  {cnt:4d}  {bar}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
def load_and_preprocess(batch_size: int = 32, verbose: bool = True):
    """Pipeline complet : chargement → nettoyage → encodage → normalisation → split.

    Returns
    -------
    loaders : dict  {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    meta    : dict  {n_features, n_classes, class_names, feature_names, scaler, le}
    """
    # ── 1. Chargement ─────────────────────────────────────────────────────────
    df = _load_raw_df()
    df = _normalize_columns(df)
    df = _clean(df)
    if verbose:
        _show_stats(df)

    # ── 2. Encodage des colonnes catégorielles ────────────────────────────────
    le_color    = LabelEncoder()
    le_spectral = LabelEncoder()
    if "Star Color" in df.columns:
        df["Star Color"]     = le_color.fit_transform(df["Star Color"].astype(str))
    if "Spectral Class" in df.columns:
        df["Spectral Class"] = le_spectral.fit_transform(df["Spectral Class"].astype(str))

    # ── 3. Features / cibles ──────────────────────────────────────────────────
    target_col   = "Star Type"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.int64)

    le = LabelEncoder()
    y  = le.fit_transform(y)
    class_names = [CLASS_LABELS.get(int(c), str(c)) for c in le.classes_]

    print(f"[Data] Features ({len(feature_cols)}) : {feature_cols}")
    print(f"[Data] Classes  ({len(class_names)}) : {class_names}")

    # ── 4. Normalisation StandardScaler ──────────────────────────────────────
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # ── 5. Split 70 / 15 / 15 ─────────────────────────────────────────────────
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

    print(f"[Data] Split → train:{len(X_tr)}  val:{len(X_val)}  test:{len(X_te)}")

    # ── 6. DataLoaders ────────────────────────────────────────────────────────
    def make_loader(X_arr, y_arr, shuffle=False):
        ds = TensorDataset(
            torch.tensor(X_arr, dtype=torch.float32),
            torch.tensor(y_arr, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    loaders = {
        "train": make_loader(X_tr,  y_tr,  shuffle=True),
        "val":   make_loader(X_val, y_val),
        "test":  make_loader(X_te,  y_te),
    }
    meta = {
        "n_features":    X.shape[1],
        "n_classes":     len(class_names),
        "class_names":   class_names,
        "feature_names": feature_cols,
        "scaler":        scaler,
        "le":            le,
    }
    return loaders, meta


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loaders, meta = load_and_preprocess()
    X_b, y_b = next(iter(loaders["train"]))
    print(f"\n[Test] Batch X : {X_b.shape}   y : {y_b.shape}")
    print("[✓] data_preparation.py OK")
