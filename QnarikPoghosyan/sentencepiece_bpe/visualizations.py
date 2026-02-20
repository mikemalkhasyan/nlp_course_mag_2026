"""
SentencePiece BPE Tokenizer — Visualizations
=============================================
Generates four publication-quality plots that illustrate how the trained
BPE vocabulary is structured and how it tokenizes Armenian text.

Plots produced:
    1. vocab_composition.png      — Donut chart: single chars / subwords / full words
    2. token_frequencies.png      — Horizontal bar chart: top-20 corpus token pieces
    3. token_length_distribution.png — Histogram: vocabulary entry length distribution
    4. sentence_tokenization.png  — Colour-coded token-block diagram for 3 test sentences
"""

import os
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import sentencepiece as spm


# ── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH  = "models/hy_bpe.model"
CORPUS_PATH = "corpus.txt"
OUT_DIR     = "visualizations"

TEST_SENTENCES = [
    "Հայաստանն ունի հարուստ պատմություն։",
    "Արհեստական բանականությունը արագ զարգանում է։",
    "Ծրագրավորումը կարևոր հմտություն է ապագայի համար։",
]

# Colour palette
PALETTE = {
    "single":  "#4C72B0",
    "subword": "#DD8452",
    "full":    "#55A868",
    "bg":      "#F8F8F8",
}

# Try to use a font that ships with matplotlib and covers Armenian Unicode
matplotlib.rcParams["font.family"]     = "DejaVu Sans"
matplotlib.rcParams["axes.spines.top"]    = False
matplotlib.rcParams["axes.spines.right"]  = False
matplotlib.rcParams["figure.facecolor"]   = PALETTE["bg"]
matplotlib.rcParams["axes.facecolor"]     = PALETTE["bg"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'. Run '1_training.py' first."
        )
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp


def categorise_vocab(sp):
    single, subword, full = [], [], []
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        n = len(piece.lstrip("▁"))
        if n == 1:
            single.append(piece)
        elif 2 <= n <= 4:
            subword.append(piece)
        else:
            full.append(piece)
    return single, subword, full


def corpus_frequencies(sp, corpus_path):
    counts = Counter()
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                counts.update(sp.encode(line, out_type=str))
    return counts


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ── Plot 1: Vocabulary Composition Donut ─────────────────────────────────────

def viz_vocab_composition(single, subword, full):
    """
    Donut chart showing the three vocabulary tiers.
    The centre annotation shows total vocab size.
    """
    sizes  = [len(single), len(subword), len(full)]
    labels = [
        f"Single chars\n(len = 1)\n{sizes[0]}",
        f"Subword fragments\n(len 2–4)\n{sizes[1]}",
        f"Full words\n(len 5+)\n{sizes[2]}",
    ]
    colors = [PALETTE["single"], PALETTE["subword"], PALETTE["full"]]
    explode = (0.03, 0.03, 0.03)

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct="%1.1f%%",
        pctdistance=0.78,
        startangle=140,
        wedgeprops=dict(width=0.52, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=11),
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
        at.set_color("white")

    # Centre label
    ax.text(0, 0, f"Vocab\n{sum(sizes)}", ha="center", va="center",
            fontsize=16, fontweight="bold", color="#333333")

    ax.set_title("Vocabulary Composition by Token Length",
                 fontsize=15, fontweight="bold", pad=20)

    fig.patch.set_facecolor(PALETTE["bg"])
    save(fig, "vocab_composition.png")


# ── Plot 2: Top-20 Token Frequencies ─────────────────────────────────────────

def viz_token_frequencies(token_counts, top_n=20):
    """
    Horizontal bar chart of the most frequent BPE pieces in the corpus.
    Bars are coloured by token-length category.
    """
    top = token_counts.most_common(top_n)
    pieces = [p for p, _ in top]
    counts = [c for _, c in top]

    # Colour each bar by category
    def bar_color(piece):
        n = len(piece.lstrip("▁"))
        if n == 1:
            return PALETTE["single"]
        elif n <= 4:
            return PALETTE["subword"]
        return PALETTE["full"]

    colors = [bar_color(p) for p in pieces]

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = range(len(pieces))
    bars = ax.barh(list(y_pos), counts, color=colors, edgecolor="white",
                   linewidth=0.8, height=0.72)

    # Value labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", ha="left", fontsize=9, color="#555555")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(pieces, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency in corpus", fontsize=12)
    ax.set_title(f"Top {top_n} Most Frequent BPE Token Pieces",
                 fontsize=15, fontweight="bold", pad=14)

    legend_patches = [
        mpatches.Patch(color=PALETTE["single"],  label="Single char (len 1)"),
        mpatches.Patch(color=PALETTE["subword"], label="Subword (len 2–4)"),
        mpatches.Patch(color=PALETTE["full"],    label="Full word (len 5+)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=10,
              framealpha=0.85)

    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    save(fig, "token_frequencies.png")


# ── Plot 3: Token Length Distribution ────────────────────────────────────────

def viz_token_length_distribution(sp):
    """
    Histogram showing how many vocabulary entries exist for each
    surface-character length (▁ stripped), with a KDE-style smooth line
    overlaid for readability.
    """
    lengths = [len(sp.id_to_piece(i).lstrip("▁")) for i in range(sp.get_piece_size())]
    max_len = max(lengths)

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = range(0, max_len + 2)

    n, bin_edges, patches = ax.hist(
        lengths, bins=list(bins), edgecolor="white", linewidth=0.8,
        color=PALETTE["subword"], alpha=0.85, rwidth=0.85,
    )

    # Colour buckets by category
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        if left_edge == 1:
            patch.set_facecolor(PALETTE["single"])
        elif 2 <= left_edge <= 4:
            patch.set_facecolor(PALETTE["subword"])
        elif left_edge >= 5:
            patch.set_facecolor(PALETTE["full"])

    # Vertical dividers
    ax.axvline(1.5, color="#999", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(4.5, color="#999", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(1,   max(n) * 0.95, "single\nchars", ha="center", va="top",
            fontsize=9, color=PALETTE["single"])
    ax.text(3.2, max(n) * 0.95, "subword\nfragments", ha="center", va="top",
            fontsize=9, color=PALETTE["subword"])
    ax.text(7,   max(n) * 0.95, "full words", ha="center", va="top",
            fontsize=9, color=PALETTE["full"])

    ax.set_xlabel("Token surface length (characters, ▁ stripped)", fontsize=12)
    ax.set_ylabel("Number of vocabulary entries", fontsize=12)
    ax.set_title("Vocabulary Entry Length Distribution",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_xticks(range(1, max_len + 1))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    save(fig, "token_length_distribution.png")


# ── Plot 4: Sentence Tokenization Diagram ────────────────────────────────────

def viz_sentence_tokenization(sp, sentences):
    """
    Colour-coded token-block diagram for each test sentence.
    Each BPE piece is rendered as a rounded rectangle; pieces that begin
    with ▁ (word-start marker) get a slightly darker shade to make word
    boundaries visible.
    """
    # Build a colour cycle per sentence
    sent_palettes = [
        ["#4C72B0", "#6B8FCC", "#2E5591", "#8AAAD8", "#1A3F75"],
        ["#DD8452", "#F0A070", "#B85A25", "#E8C4A0", "#8A3A10"],
        ["#55A868", "#78C48A", "#357A47", "#A0D8AA", "#1A5228"],
    ]

    fig, axes = plt.subplots(len(sentences), 1,
                             figsize=(13, 3.2 * len(sentences)))
    fig.subplots_adjust(hspace=0.55)

    for ax, sentence, palette in zip(axes, sentences, sent_palettes):
        pieces = sp.encode(sentence, out_type=str)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Measure widths proportional to piece length
        raw_widths = [max(len(p.lstrip("▁")), 1) for p in pieces]
        total = sum(raw_widths)
        gap = 0.008
        available = 1.0 - gap * (len(pieces) - 1)

        x = 0.0
        box_h = 0.55
        box_y = 0.22

        for idx, (piece, rw) in enumerate(zip(pieces, raw_widths)):
            w = available * (rw / total)
            is_word_start = piece.startswith("▁")
            color = palette[idx % len(palette)]
            # Darken word-start pieces slightly
            if is_word_start:
                color = palette[min(idx % len(palette) + 1, len(palette) - 1)]

            # Rounded box
            fancy = mpatches.FancyBboxPatch(
                (x, box_y), w, box_h,
                boxstyle="round,pad=0.01",
                linewidth=1.2,
                edgecolor="white",
                facecolor=color,
                alpha=0.92,
                transform=ax.transAxes,
            )
            ax.add_patch(fancy)

            # Token text
            display = piece.replace("▁", "")
            ax.text(x + w / 2, box_y + box_h / 2, display,
                    ha="center", va="center",
                    fontsize=max(7, min(11, int(200 / len(pieces)))),
                    fontweight="bold", color="white",
                    transform=ax.transAxes,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

            # Token ID below
            token_id = sp.piece_to_id(piece)
            ax.text(x + w / 2, box_y - 0.06, str(token_id),
                    ha="center", va="top", fontsize=7,
                    color="#666666", transform=ax.transAxes)

            x += w + gap

        # Sentence header
        short = sentence if len(sentence) <= 55 else sentence[:52] + "…"
        ax.set_title(f"{short}   [{len(pieces)} tokens]",
                     fontsize=10.5, pad=6, loc="left", color="#333333")

    fig.suptitle("BPE Tokenization of Armenian Test Sentences",
                 fontsize=14, fontweight="bold", y=1.01)

    save(fig, "sentence_tokenization.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("SENTENCEPIECE BPE — VISUALIZATIONS")
    print("=" * 60 + "\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    sp           = load_model(MODEL_PATH)
    single, subword, full = categorise_vocab(sp)
    token_counts = corpus_frequencies(sp, CORPUS_PATH)

    print("Generating plots …")
    viz_vocab_composition(single, subword, full)
    viz_token_frequencies(token_counts, top_n=20)
    viz_token_length_distribution(sp)
    viz_sentence_tokenization(sp, TEST_SENTENCES)

    print(f"\n✓ All visualizations saved to '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
