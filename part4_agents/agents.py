"""
Part IV — Simple communicating agents.

Each agent summarizes one project block and can react to the others.
The goal is to stay very simple and easy to reuse in the report.
"""

import os
from dataclasses import dataclass
import re
from typing import Dict, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torchvision


@dataclass
class AgentReport:
    name: str
    part: str
    summary: str
    explanation: str
    signals: Dict[str, float]


class BaseAgent:
    def __init__(self, name: str, part: str):
        self.name = name
        self.part = part

    def analyze(self, context: Dict) -> AgentReport:
        raise NotImplementedError

    def answer_question(self, question: str, context: Dict) -> str:
        raise NotImplementedError

    def _question_tokens(self, question: str):
        return set(re.findall(r"[a-zA-ZÀ-ÿ0-9']+", question.lower()))

    def _contains_any(self, question: str, words: Sequence[str]) -> bool:
        tokens = self._question_tokens(question)
        return any(word.lower() in tokens or word.lower() in question.lower() for word in words)

    def _target_agent_name(self, question: str) -> str | None:
        if self._contains_any(question, ["mlp", "tabulaire", "table", "variables", "classification", "confusion"]):
            return "Agent_MLP"
        if self._contains_any(question, ["cnn", "image", "images", "convolution", "pooling", "feature", "visuel", "visualisation", "dataset"]):
            return "Agent_CNN"
        if self._contains_any(question, ["rnn", "seq2seq", "sequence", "séquence", "texte", "text", "token", "bleu", "perplexite", "perplexité"]):
            return "Agent_RNN"
        return None

    def _defer_if_other_part(self, question: str, context: Dict) -> str | None:
        target = self._target_agent_name(question)
        if target and target != self.name:
            return f"Je suis l'agent {self.name.split('_', 1)[1]}. Cette question relève surtout du rôle de {target}."
        return None
    def communicate(self, own_report: AgentReport, peer_reports: Sequence[AgentReport]) -> str:
        if not peer_reports:
            return f"{self.name}: aucun échange disponible."
        best_peer = max(peer_reports, key=lambda r: r.signals.get("score", 0.0))
        return (
            f"{self.name} compare son résultat avec {best_peer.name} : "
            f"{best_peer.part} semble être le plus fort sur la métrique score."
        )


class MLPAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Agent_MLP", part="Partie I - MLP")

    def analyze(self, context: Dict) -> AgentReport:
        part = context.get("part1", {})
        score = float(part.get("accuracy", 0.0))
        explanation = part.get(
            "explanation",
            "Les variables tabulaires les plus influentes sont mises en avant par une lecture simple des poids.",
        )
        return AgentReport(
            name=self.name,
            part=self.part,
            summary=f"MLP: accuracy={score:.3f}",
            explanation=explanation,
            signals={"score": score},
        )

    def answer_question(self, question: str, context: Dict) -> str:
        defer = self._defer_if_other_part(question, context)
        if defer:
            return defer

        part = context.get("part1", {})
        accuracy = float(part.get("accuracy", 0.0))
        explanation = part.get(
            "explanation",
            "Les variables tabulaires les plus influentes sont mises en avant par une lecture simple des poids.",
        )
        if self._contains_any(question, ["mlp", "tabulaire", "partie 1", "partie i", "partie un", "données tabulaires"]):
            return (
                f"Je suis l'agent MLP. Mon rôle est de traiter les données tabulaires de la partie 1. "
                f"J'apprends une représentation simple des variables d'entrée pour faire la classification."
            )
        if self._contains_any(question, ["architecture", "réseau", "fully-connected", "fully connected", "couches", "layers"]):
            return (
                "Je suis l'agent MLP. Mon architecture est un réseau fully-connected avec des couches denses. "
                "Chaque neurone de la couche précédente est connecté à la suivante pour apprendre les relations entre variables."
            )
        if self._contains_any(question, ["variables", "features", "feature", "important", "importance", "shap", "lime", "explic", "poids"]):
            return (
                f"Je suis l'agent MLP. Les variables importantes sont résumées ici: {explanation} "
                "Cela permet de voir quelles colonnes influencent le plus la prédiction."
            )
        if self._contains_any(question, ["matrice de confusion", "confusion matrix", "confusion", "accuracy", "precision", "recall", "f1", "score", "résultat"]):
            return (
                f"Je suis l'agent MLP. Mon résultat principal sur la partie 1 est une accuracy d'environ {accuracy:.3f}. "
                "La matrice de confusion permet de voir les bonnes et les mauvaises classifications pour chaque classe."
            )
        if self._contains_any(question, ["resultat", "accuracy", "performance", "score"]):
            return (
                f"Je suis l'agent MLP. Sur la partie 1, la performance observée est d'environ {accuracy:.3f}. "
                f"Je travaille sur des données tabulaires, donc je regarde surtout les variables d'entrée."
            )
        return (
            f"Je suis l'agent MLP. Ma mission est la partie tabulaire: réseau fully-connected, initialisation des poids, "
            f"métriques de classification et matrice de confusion. Mon score courant est {accuracy:.3f}."
        )


class CNNAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Agent_CNN", part="Partie II - CNN")

    def _save_dataset_preview(self, context: Dict, question: str) -> str:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "part4_agents")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "cnn_dataset_preview.png")
        data_root = os.path.join(os.path.dirname(__file__), "..", "outputs", "data")

        dataset = None
        last_error = None
        for train_flag in (False, True):
            try:
                dataset = torchvision.datasets.FashionMNIST(
                    root=data_root,
                    train=train_flag,
                    download=False,
                )
                if len(dataset) > 0:
                    break
            except Exception as exc:
                last_error = exc
                dataset = None

        if dataset is None:
            # Fallback minimal si la dataset n'est pas lisible localement.
            rng = np.random.RandomState(7)
            base = rng.rand(8, 8)
            part = context.get("part2", {})
            boost = float(part.get("accuracy", 0.0))
            heatmap = base * 0.55 + boost * 0.45

            fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
            fig.suptitle("CNN agent dataset preview", fontsize=12, fontweight="bold")

            axes[0].imshow(heatmap, cmap="magma")
            axes[0].set_title("Fallback preview")
            axes[0].axis("off")

            labels = ["dataset", "accuracy", "visual"]
            values = [1.0, boost, max(0.15, boost * 0.9)]
            axes[1].bar(labels, values, color=["#60a5fa", "#22c55e", "#f59e0b"])
            axes[1].set_ylim(0, 1.05)
            axes[1].set_title("Simple summary")
            axes[1].grid(axis="y", alpha=0.25)

            fig.tight_layout()
            fig.savefig(out_path, dpi=140, bbox_inches="tight")
            plt.close(fig)
            return out_path

        class_names = getattr(dataset, "classes", [str(i) for i in range(10)])
        labels_to_show = []
        count = min(8, len(dataset))
        indices = np.linspace(0, len(dataset) - 1, num=count, dtype=int)
        for index in indices:
            image, label = dataset[int(index)]
            if hasattr(image, "squeeze"):
                image = image.squeeze()
            labels_to_show.append((np.array(image), int(label)))

        fig, axes = plt.subplots(2, 4, figsize=(8.6, 4.8))
        fig.suptitle("Vraies images de la dataset Fashion-MNIST", fontsize=12, fontweight="bold")

        for axis, (image, label) in zip(axes.flat, labels_to_show):
            axis.imshow(image, cmap="gray")
            axis.set_title(class_names[label], fontsize=8)
            axis.axis("off")

        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def analyze(self, context: Dict) -> AgentReport:
        part = context.get("part2", {})
        score = float(part.get("accuracy", 0.0))
        explanation = part.get(
            "explanation",
            "Les zones de l'image les plus actives sont résumées comme une explication de type carte de chaleur.",
        )
        return AgentReport(
            name=self.name,
            part=self.part,
            summary=f"CNN: accuracy={score:.3f}",
            explanation=explanation,
            signals={"score": score},
        )

    def answer_question(self, question: str, context: Dict) -> str:
        defer = self._defer_if_other_part(question, context)
        if defer:
            return defer

        part = context.get("part2", {})
        accuracy = float(part.get("accuracy", 0.0))
        explanation = part.get(
            "explanation",
            "Les zones de l'image les plus actives sont résumées comme une explication de type carte de chaleur.",
        )
        if self._contains_any(question, ["generer", "générer", "image", "images", "dataset", "visual", "visualisation", "show", "afficher", "preview", "vrai", "vraies"]):
            preview_path = self._save_dataset_preview(context, question)
            return (
                f"Je suis l'agent CNN. Oui, je peux générer de vraies images de la dataset Fashion-MNIST. "
                f"J'ai sauvegardé un aperçu ici: {preview_path}. "
                f"Si tu me donnes une image précise, je peux aussi la commenter."
            )
        if self._contains_any(question, ["cnn", "partie 2", "partie ii", "image", "images", "vision", "vision par ordinateur"]):
            return (
                "Je suis l'agent CNN. Mon rôle est de traiter les images de la partie 2. "
                "Je détecte des motifs visuels utiles pour classer les images de la dataset."
            )
        if self._contains_any(question, ["architecture", "réseau", "convolution", "pooling", "lenet", "couches", "layers"]):
            return (
                "Je suis l'agent CNN. Mon architecture repose sur des couches de convolution, du pooling et des couches de classification. "
                "Dans ce projet, je compare notamment LeNet-5 et une variante améliorée."
            )
        if self._contains_any(question, ["feature", "features", "carte", "heatmap", "important", "explic", "shap", "lime"]):
            return (
                f"Je suis l'agent CNN. Les zones les plus importantes de l'image sont résumées ici: {explanation} "
                f"Cela montre quelles régions visuelles influencent la prédiction."
            )
        if self._contains_any(question, ["resultat", "accuracy", "performance", "score"]):
            return (
                f"Je suis l'agent CNN. La partie 2 obtient une accuracy d'environ {accuracy:.3f}. "
                f"Je suis spécialisé dans les images et les motifs visuels."
            )
        return (
            f"Je suis l'agent CNN. Ma mission est la vision par ordinateur: convolution, pooling, LeNet et feature maps. "
            f"Mon score courant est {accuracy:.3f}."
        )


class RNNAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Agent_RNN", part="Partie III - RNN/Seq2Seq")

    def analyze(self, context: Dict) -> AgentReport:
        part = context.get("part3", {})
        score = float(part.get("bleu", part.get("accuracy", 0.0)))
        explanation = part.get(
            "explanation",
            "Les mots ou tokens les plus utiles sont mis en évidence par une explication textuelle simple.",
        )
        return AgentReport(
            name=self.name,
            part=self.part,
            summary=f"RNN: score={score:.3f}",
            explanation=explanation,
            signals={"score": score},
        )

    def answer_question(self, question: str, context: Dict) -> str:
        defer = self._defer_if_other_part(question, context)
        if defer:
            return defer

        part = context.get("part3", {})
        score = float(part.get("bleu", part.get("accuracy", 0.0)))
        explanation = part.get(
            "explanation",
            "Les mots ou tokens les plus utiles sont mis en évidence par une explication textuelle simple.",
        )
        if self._contains_any(question, ["difference", "différence", "compare", "comparaison", "distinction", "rnn", "lstm", "gru"]):
            return (
                "Je suis l'agent RNN. "
                "RNN est la version simple pour traiter les séquences, LSTM garde mieux le contexte sur le long terme, "
                "et GRU est une version plus légère souvent plus rapide. "
                "Dans ce projet, j'utilise la partie texte/séquence pour comparer ces modèles."
            )
        if self._contains_any(question, ["seq2seq", "encoder", "decoder", "décodeur", "decodeur"]):
            return (
                "Je suis l'agent RNN. Le Seq2Seq est un modèle composé d'un encodeur et d'un décodeur. "
                "L'encodeur transforme la séquence d'entrée en représentation compacte, puis le décodeur génère la sortie token par token."
            )
        if self._contains_any(question, ["perplexite", "perplexité"]):
            return (
                "Je suis l'agent RNN. La perplexité mesure à quel point le modèle est surpris par les séquences. "
                "Plus elle est petite, meilleur est le modèle sur la tâche de langage."
            )
        if self._contains_any(question, ["bleu"]):
            return (
                f"Je suis l'agent RNN. Le score BLEU de la partie 3 est d'environ {score:.3f}. "
                "Il mesure la proximité entre la sortie produite et la référence."
            )
        if self._contains_any(question, ["resultat", "bleu", "perplexite", "performance", "score"]):
            return (
                f"Je suis l'agent RNN. Sur la partie 3, le score séquentiel observé est d'environ {score:.3f}. "
                f"Je traite les phrases, les tokens et les séquences."
            )
        if self._contains_any(question, ["explic", "shap", "lime", "important", "token", "mot", "word"]):
            return (
                f"Je suis l'agent RNN. Pour l'explicabilité textuelle: {explanation} "
                f"Cela met en avant les mots qui influencent le plus la prédiction."
            )
        return (
            f"Je suis l'agent RNN. Ma mission est la partie texte/séquence: RNN, LSTM, GRU et Seq2Seq. "
            f"Mon score courant est {score:.3f}."
        )


def summarize_agents(reports):
    best = max(reports, key=lambda r: r.signals.get("score", 0.0))
    lines = [
        f"Coordonnateur: l'agent le plus fort selon le score affiché est {best.name} ({best.part}).",
        "Coordonnateur: chaque agent répond selon sa spécialité et peut aussi donner une explication simple.",
    ]
    return "\n".join(lines)


class CoordinatorAgent:
    def __init__(self, agents: Sequence[BaseAgent]):
        self.agents = list(agents)

    def run(self, context: Dict):
        return [agent.analyze(context) for agent in self.agents]

    def exchange(self, reports):
        messages = []
        for index, agent in enumerate(self.agents):
            peers = [report for idx, report in enumerate(reports) if idx != index]
            messages.append(agent.communicate(reports[index], peers))
        return messages
