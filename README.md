# Projet Deep Learning — EMSI 2025-2026
## Module : Deep Learning | Département Informatique

Projet individuel complet couvrant **3 paradigmes de deep learning** en PyTorch :
tabular (MLP), images (CNN), séquences (RNN/LSTM/GRU/Seq2Seq).

Une partie simple d'**agents communicants** est aussi ajoutée pour relier les 3 blocs avec une explication légère de type SHAP/LIME.

---

## 📁 Structure du projet

```
Deeplearning_project/
├── requirements.txt
├── README.md
├── outputs/           ← Graphiques et résultats générés automatiquement
│   ├── part1/
│   ├── part2/
│   └── part3/
├── part1_mlp/         ← Partie I : MLP sur données tabulaires
│   ├── data_preparation.py
│   ├── mlp_sequential.py
│   ├── mlp_custom.py
│   ├── weight_init.py
│   ├── train_evaluate.py
│   └── main.py
├── part2_cnn/         ← Partie II : CNN & Vision par ordinateur
│   ├── manual_ops.py
│   ├── lenet.py
│   ├── experiments.py
│   ├── visualize.py
│   └── main.py
└── part3_rnn/         ← Partie III : RNN/LSTM/GRU & Seq2Seq
    ├── models.py
    ├── seq2seq.py
    ├── decoding.py
    ├── data_prep.py
    ├── train_evaluate.py
    └── main.py
└── part4_agents/      ← Partie IV : agents simples + interface Tkinter + explicabilité
    ├── agents.py
    ├── explainability.py
    └── main.py
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Exécution

### Partie I — MLP (30 pts)
```bash
python part1_mlp/main.py
```
**Dataset :** Star Classification (propriétés physiques et spectrales des étoiles)  
**Sorties :**
- `outputs/part1/curves_seq_vs_custom.png` — courbes d'apprentissage
- `outputs/part1/curves_weight_init.png`  — comparaison des initialisations
- `outputs/part1/cm_*.png`                — matrices de confusion (**imprimées dans le terminal**)

---

### Partie II — CNN (35 pts)
```bash
python part2_cnn/main.py
```
**Dataset :** Fashion-MNIST (téléchargé automatiquement)  
**Sorties :**
- `outputs/part2/cnn_curves.png`         — courbes LeNet-5 vs Improved LeNet
- `outputs/part2/cnn_experiments.png`    — impact padding/stride/filtres
- `outputs/part2/featuremap_*.png`       — **cartes de features rendues** (forward hooks)

---

### Partie III — RNN/LSTM/GRU & Seq2Seq (35 pts)
```bash
python part3_rnn/main.py
```
**Dataset :** IMDB Sentiment (ou données synthétiques en fallback)  
**Sorties :**
- `outputs/part3/rnn_comparison.png` — courbes RNN vs LSTM vs GRU
- **Perplexité et score BLEU imprimés dans le terminal** (`[✓]`)

### Partie IV — Agents communicants + explicabilité simple
```bash
python part4_agents/main.py
```
**Rôle :**
- un agent pour la partie MLP
- un agent pour la partie CNN
- un agent pour la partie RNN
- un coordinateur qui échange les messages entre eux
**Interface :**
- un champ de saisie pour poser la question
- 3 zones de réponse affichées en parallèle
- chaque agent répond selon sa partie
**Explicabilité :**
- version simple inspirée de SHAP/LIME
- mise en avant des variables, filtres ou mots les plus influents
- sortie textuelle facile à intégrer dans le rapport

---

## 📋 Ce qui est implémenté

| Partie | Contenu | Points |
|--------|---------|--------|
| **I** | MLP Sequential + Custom `nn.Module`, 3 stratégies d'initialisation (Gaussian/Constant/Xavier), métriques complètes (accuracy, precision, recall, F1, confusion matrix) | 30 |
| **II** | Convolution 2D manuelle (numpy), max-pooling & avg-pooling manuels, formule des dimensions, LeNet-5, LeNet amélioré, expériences hyperparamètres, visualisation de feature maps | 35 |
| **III** | RNN vanilla (cellule manuelle), LSTM, GRU, Seq2Seq encoder-décodeur, Greedy decoding, Beam Search decoding, gradient clipping (BPTT), Perplexité, BLEU | 35 |
| **IV** | 3 agents simples (MLP/CNN/RNN) + communication + explicabilité textuelle inspirée de SHAP/LIME | bonus |

---

## ✅ Vérifications manuelles

- [x] **Partie I** — Matrice de confusion imprimée dans le terminal + sauvegardée
- [x] **Partie II** — Feature maps rendues et sauvegardées (`outputs/part2/featuremap_*.png`)
- [x] **Partie III** — Perplexité et score BLEU affichés dans le terminal (`[✓]`)

---

## 🔬 Concepts clés abordés

- `nn.Module`, `parameters()`, `state_dict()`
- Initialisations : Xavier, Gaussienne, Constante
- Cross-corrélation 2D manuelle, pooling manuel
- LeNet-5 et variante améliorée
- Forward hooks pour visualisation
- RNN cellulaire, LSTM, GRU
- BPTT + gradient clipping
- Encoder-Decoder Seq2Seq
- Greedy vs Beam Search decoding
- Perplexité & BLEU score

---

*EMSI Casablanca — Département Informatique — Deep Learning 2025-2026*
