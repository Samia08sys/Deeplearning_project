"""
Part IV — Three agent dashboard.

Dashboard layout:
  1. Question commune : petite zone centrée en haut.
  2. Les 3 cartes d'agents (MLP / CNN / RNN) juste en dessous.
  3. Coordonnateur : petit résumé centré, en bas des 3 cartes.

Le corps de la fenêtre est scrollable (Canvas + Scrollbar) pour garantir
que le coordonnateur reste toujours accessible, même sur un petit écran.
Le logo de chaque agent est un robot dessiné en vectoriel (Canvas), dans
la couleur d'accent de la carte — pas une image importée.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk
import re
import math

sys.path.insert(0, os.path.dirname(__file__))

from agents import CNNAgent, MLPAgent, RNNAgent, summarize_agents
from explainability import explain_image, explain_tabular, explain_text


def build_demo_context():
    return {
        "part1": {
            "accuracy": 0.91,
            "explanation": explain_tabular(
                ["mass", "temperature", "luminosity", "radius"],
                [0.82, -0.21, 0.67, 0.14],
            ),
        },
        "part2": {
            "accuracy": 0.89,
            "explanation": explain_image(
                ["edge_1", "edge_2", "texture_1", "texture_2"],
                [0.44, 0.31, 0.28, 0.19],
            ),
        },
        "part3": {
            "bleu": 0.42,
            "explanation": explain_text(
                ["good", "movie", "not", "bad", "story", "actor"],
                [0.16, 0.12, -0.09, 0.08, 0.07, 0.05],
            ),
        },
    }


class ChatbotLogo(tk.Canvas):
    """Logo robot/chatbot dessiné en vectoriel (pas une photo collée).

    Une tête-écran arrondie avec deux yeux, un sourire et une antenne,
    dans la couleur d'accent de la carte — lisible à toutes les tailles.
    """

    def __init__(self, parent, size=48, accent="#60a5fa", bg="#111827"):
        super().__init__(parent, width=size, height=size, bg=bg, highlightthickness=0)
        self._draw(size, accent, bg)

    @staticmethod
    def _rounded_rect(canvas, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1 + r, y1,
            x2 - r, y1,
            x2, y1,
            x2, y1 + r,
            x2, y2 - r,
            x2, y2,
            x2 - r, y2,
            x1 + r, y2,
            x1, y2,
            x1, y2 - r,
            x1, y1 + r,
            x1, y1,
        ]
        return canvas.create_polygon(points, smooth=True, **kwargs)

    def _draw(self, s, accent, bg):
        face = "#e8f2ff"  # couleur claire pour les yeux / sourire / antenne

        # Antenne.
        self.create_line(s * 0.50, s * 0.10, s * 0.50, s * 0.20, fill=accent, width=max(1, int(s * 0.035)))
        self.create_oval(s * 0.44, s * 0.04, s * 0.56, s * 0.16, fill=accent, outline="")

        # Oreilles / haut-parleurs latéraux.
        self._rounded_rect(self, s * 0.06, s * 0.40, s * 0.18, s * 0.62, s * 0.05, fill=accent, outline="")
        self._rounded_rect(self, s * 0.82, s * 0.40, s * 0.94, s * 0.62, s * 0.05, fill=accent, outline="")

        # Tête.
        head = (s * 0.16, s * 0.20, s * 0.84, s * 0.88)
        self._rounded_rect(self, *head, s * 0.16, fill=accent, outline="")

        # Écran / visage (inset plus sombre).
        inset = (s * 0.26, s * 0.32, s * 0.74, s * 0.76)
        self._rounded_rect(self, *inset, s * 0.09, fill=bg, outline="")

        ix1, iy1, ix2, iy2 = inset
        iw, ih = ix2 - ix1, iy2 - iy1

        # Yeux.
        eye_r = iw * 0.085
        eye_y = iy1 + ih * 0.38
        for ex in (ix1 + iw * 0.30, ix1 + iw * 0.70):
            self.create_oval(ex - eye_r, eye_y - eye_r, ex + eye_r, eye_y + eye_r, fill=face, outline="")

        # Sourire.
        self.create_arc(
            ix1 + iw * 0.22, iy1 + ih * 0.40, ix2 - iw * 0.22, iy1 + ih * 0.92,
            start=200, extent=140, style="arc", outline=face, width=max(1, int(s * 0.028)),
        )


class AgentCard:
    def __init__(self, parent, agent, accent, badge_color, intro_text):
        self.agent = agent
        self.frame = tk.Frame(parent, bg="#111827", padx=10, pady=10, highlightthickness=2, highlightbackground=accent)

        top = tk.Frame(self.frame, bg="#111827")
        top.pack(fill="x")

        ChatbotLogo(top, size=44, accent=accent).pack(side="left", padx=(0, 10), anchor="n")

        title_box = tk.Frame(top, bg="#111827")
        title_box.pack(side="left", fill="x", expand=True)

        name_row = tk.Frame(title_box, bg="#111827")
        name_row.pack(fill="x")

        tk.Label(
            name_row,
            text=agent.name,
            bg="#111827",
            fg="white",
            font=("Segoe UI", 14, "bold"),
        ).pack(side="left", anchor="w")

        badge = tk.Label(
            name_row,
            text=agent.part,
            bg=badge_color,
            fg="#111827",
            font=("Segoe UI", 8, "bold"),
            padx=8,
            pady=3,
        )
        badge.pack(side="right", anchor="e")

        tk.Label(
            title_box,
            text=intro_text,
            bg="#111827",
            fg="#cbd5e1",
            font=("Segoe UI", 8),
            wraplength=260,
            justify="left",
        ).pack(anchor="w", fill="x", pady=(3, 0))

        meta = tk.Frame(self.frame, bg="#111827")
        meta.pack(fill="x", pady=(8, 6))
        self.score_label = tk.Label(meta, text="Score: --", bg="#111827", fg=accent, font=("Segoe UI", 9, "bold"))
        self.score_label.pack(side="left")
        self.context_label = tk.Label(meta, text="Contexte: prêt", bg="#111827", fg="#94a3b8", font=("Segoe UI", 8))
        self.context_label.pack(side="right")

        self.response = tk.Text(
            self.frame,
            height=6,
            wrap="word",
            font=("Segoe UI", 9),
            bg="#0b1220",
            fg="#e5e7eb",
            insertbackground="white",
            relief="flat",
            padx=8,
            pady=8,
        )
        self.response.pack(fill="both", expand=True)
        self.response.configure(state="disabled")
        self.set_message("La réponse de l'agent apparaîtra ici.")

        self.preview_frame = tk.Frame(self.frame, bg="#111827")
        self.preview_frame.pack(fill="x", pady=(8, 0))
        self.preview_label = tk.Label(
            self.preview_frame,
            text="",
            bg="#111827",
            fg="#cbd5e1",
            font=("Segoe UI", 8),
            justify="center",
        )
        self.preview_label.pack(anchor="center")
        self.preview_image_label = tk.Label(self.preview_frame, bg="#111827")
        self.preview_image_label.pack(anchor="center", pady=(4, 0))
        self._preview_photo = None

    def set_message(self, text):
        self.response.configure(state="normal")
        self.response.delete("1.0", "end")
        self.response.insert("1.0", text)
        self.response.configure(state="disabled")

    def set_status(self, score_text: str, context_text: str):
        self.score_label.configure(text=score_text)
        self.context_label.configure(text=context_text)

    def set_preview(self, image_path: str | None, caption: str = ""):
        if not image_path or not os.path.exists(image_path):
            self.preview_label.configure(text="")
            self.preview_image_label.configure(image="")
            self._preview_photo = None
            return

        try:
            photo = tk.PhotoImage(file=image_path)
            max_width = 280
            max_height = 160
            width = max(1, photo.width())
            height = max(1, photo.height())
            scale_x = max(1, math.ceil(width / max_width))
            scale_y = max(1, math.ceil(height / max_height))
            if scale_x > 1 or scale_y > 1:
                photo = photo.subsample(scale_x, scale_y)
            self._preview_photo = photo
            self.preview_image_label.configure(image=photo)
            self.preview_label.configure(text=caption)
        except tk.TclError:
            self.preview_label.configure(text="Aperçu image non affichable sur cette plateforme.")
            self.preview_image_label.configure(image="")
            self._preview_photo = None

    def clear_preview(self):
        self.set_preview(None)


class AgentApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Deep Learning Agents")
        self.root.geometry("1320x820")
        self.root.minsize(1100, 700)
        try:
            # Maximise la fenêtre au démarrage (Windows) pour que tout
            # (notamment le coordonnateur) soit visible sans avoir à scroller.
            self.root.state("zoomed")
        except tk.TclError:
            # Sur les plateformes où "zoomed" n'existe pas (ex. macOS),
            # on retombe sur la géométrie fixe + le scroll reste disponible.
            if sys.platform.startswith("linux"):
                try:
                    self.root.attributes("-zoomed", True)
                except tk.TclError:
                    pass
        self.root.configure(bg="#0f172a")

        self.context = build_demo_context()
        self.agents = [MLPAgent(), CNNAgent(), RNNAgent()]
        self.cards = {}
        self._build_ui()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.configure("Secondary.TButton", font=("Segoe UI", 9), padding=8)

        # ---------- En-tête ----------
        header = tk.Frame(self.root, bg="#111827", padx=28, pady=16)
        header.pack(fill="x")

        left = tk.Frame(header, bg="#111827")
        left.pack(side="left", fill="x", expand=True)
        tk.Label(
            left,
            text="Interface des 3 agents",
            font=("Segoe UI", 22, "bold"),
            fg="white",
            bg="#111827",
        ).pack(anchor="w")
        tk.Label(
            left,
            text="Trois agents spécialisés répondent chacun sur sa partie: MLP, CNN et RNN/Seq2Seq.",
            font=("Segoe UI", 10),
            fg="#cbd5e1",
            bg="#111827",
        ).pack(anchor="w", pady=(4, 0))

        ChatbotLogo(header, size=90, accent="#60a5fa", bg="#111827").pack(side="right", padx=(16, 0))

        # ---------- Corps (scrollable, pour que rien ne soit jamais coupé) ----------
        container = tk.Frame(self.root, bg="#0f172a")
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, bg="#0f172a", highlightthickness=0)
        # Pas de barre de défilement visible : le scroll (si jamais
        # nécessaire) se fait uniquement via la molette de la souris.
        vscroll = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.pack(side="left", fill="both", expand=True)

        body = tk.Frame(canvas, bg="#0f172a", padx=22, pady=14)
        body_id = canvas.create_window((0, 0), window=body, anchor="nw")

        def _update_scrollregion(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfig(body_id, width=event.width)
            _update_scrollregion()

        body.bind("<Configure>", _update_scrollregion)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # 1) Zone "question commune" : petite, centrée en haut.
        question_wrap = tk.Frame(body, bg="#0f172a")
        question_wrap.pack(fill="x")

        input_card = tk.Frame(question_wrap, bg="#111827", padx=14, pady=12, highlightthickness=1, highlightbackground="#334155")
        input_card.pack(pady=(0, 14))  # pas de fill -> reste centré et compact

        tk.Label(
            input_card,
            text="Question commune envoyée aux 3 agents",
            bg="#111827",
            fg="white",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="center")

        self.question_var = tk.StringVar(value="Explique ton rôle et donne ton résultat principal.")
        entry = tk.Entry(
            input_card,
            textvariable=self.question_var,
            font=("Segoe UI", 12),
            width=58,
            relief="flat",
            bd=0,
            highlightthickness=2,
            highlightbackground="#334155",
            highlightcolor="#60a5fa",
            insertbackground="white",
            bg="#1e293b",
            fg="white",
            justify="center",
        )
        entry.pack(ipady=7, pady=(8, 10))
        entry.focus_set()

        actions = tk.Frame(input_card, bg="#111827")
        actions.pack()
        ttk.Button(actions, text="Envoyer aux 3 agents", style="Primary.TButton", command=self.ask_agents).pack(side="left")
        ttk.Button(actions, text="Question demo", style="Secondary.TButton", command=self.load_demo_question).pack(side="left", padx=6)
        ttk.Button(actions, text="Réinitialiser", style="Secondary.TButton", command=self.reset_cards).pack(side="left")

        # 2) Les 3 cartes d'agents, sous la question.
        cards_row = tk.Frame(body, bg="#0f172a")
        cards_row.pack(fill="both", expand=True)

        definitions = [
            (
                self.agents[0],
                "#3b82f6",
                "#dbeafe",
                "Agent spécialisé sur les données tabulaires, les variables d'entrée et les métriques de classification.",
            ),
            (
                self.agents[1],
                "#22c55e",
                "#dcfce7",
                "Agent spécialisé sur les images, la convolution, le pooling et les cartes de caractéristiques.",
            ),
            (
                self.agents[2],
                "#f59e0b",
                "#fef3c7",
                "Agent spécialisé sur les séquences, le texte, le BLEU et la perplexité.",
            ),
        ]

        for column, (agent, accent, badge_color, intro) in enumerate(definitions):
            card = AgentCard(cards_row, agent, accent, badge_color, intro)
            card.frame.grid(row=0, column=column, sticky="nsew", padx=8, pady=8)
            cards_row.grid_columnconfigure(column, weight=1)
            cards_row.grid_rowconfigure(0, weight=1)
            self.cards[agent.name] = card

        # 3) Coordonnateur : petit résumé centré, en bas des 3 cartes.
        coordinator_wrap = tk.Frame(body, bg="#0f172a")
        coordinator_wrap.pack(fill="x")

        summary = tk.Frame(coordinator_wrap, bg="#1d4ed8", padx=14, pady=10)
        summary.pack(pady=(14, 0))  # pas de fill -> reste centré et compact

        tk.Label(
            summary,
            text="Coordonnateur",
            bg="#1d4ed8",
            fg="white",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="center")
        self.summary_label = tk.Label(
            summary,
            text="Pose une question pour voir le résumé global.",
            bg="#1d4ed8",
            fg="white",
            font=("Segoe UI", 10),
            justify="center",
            anchor="center",
            wraplength=800,
        )
        self.summary_label.pack(pady=(6, 0))

    def load_demo_question(self):
        self.question_var.set("Quel est ton rôle, ton résultat principal et ton explication ?")

    def reset_cards(self):
        self.summary_label.configure(text="Pose une question pour voir le résumé global.")
        for agent in self.agents:
            card = self.cards[agent.name]
            card.set_message("La réponse de l'agent apparaîtra ici.")
            card.set_status("Score: --", "Contexte: prêt")
            card.clear_preview()

    def ask_agents(self):
        question = self.question_var.get().strip() or "Explique ton rôle et ton résultat principal."
        reports = []

        for agent in self.agents:
            card = self.cards[agent.name]
            response = agent.answer_question(question, self.context)
            card.set_message(response)

            if agent.name == "Agent_MLP":
                score = float(self.context["part1"]["accuracy"])
                context_text = "Tabulaire | MLP | variables + confusion matrix"
            elif agent.name == "Agent_CNN":
                score = float(self.context["part2"]["accuracy"])
                context_text = "Image | CNN | convolution + feature maps"
            else:
                score = float(self.context["part3"].get("bleu", 0.0))
                context_text = "Séquence | RNN | BLEU + perplexité"

            card.set_status(f"Score: {score:.3f}", context_text)

            if agent.name == "Agent_CNN":
                match = re.search(r"ici:\s*(.+?\.png)", response)
                if match:
                    image_path = match.group(1)
                    card.set_preview(image_path, caption="Aperçu réel de la dataset généré par le CNN agent")
                else:
                    card.clear_preview()
            else:
                card.clear_preview()

            reports.append(type("Report", (), {"name": agent.name, "part": agent.part, "signals": {"score": score}})())

        self.summary_label.configure(text=summarize_agents(reports))


def main():
    root = tk.Tk()
    AgentApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()