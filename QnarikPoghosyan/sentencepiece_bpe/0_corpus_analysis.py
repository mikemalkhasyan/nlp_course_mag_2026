"""
Corpus Analysis - Armenian text
Analyzes corpus.txt for word frequency, rare words, punctuation/symbol stats, etc.
"""

import sys
import io
import re
from collections import Counter

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CORPUS_PATH = "corpus.txt"

# ── Load ──────────────────────────────────────────────────────────────────────
with open(CORPUS_PATH, encoding="utf-8") as f:
    raw_text = f.read()

lines = [l for l in raw_text.splitlines() if l.strip()]
print(f"Total lines (non-empty): {len(lines)}")
print(f"Total characters:        {len(raw_text)}")

# ── Tokenise into words ───────────────────────────────────────────────────────
# Keep Armenian letters, digits, and hyphens inside words
word_pattern = re.compile(r"[\u0531-\u0587a-zA-Z0-9]+(?:[-՝][\u0531-\u0587a-zA-Z0-9]+)*")
words_raw = word_pattern.findall(raw_text)
words = [w.lower() for w in words_raw]

print(f"\nTotal word tokens:       {len(words)}")
print(f"Unique words (types):    {len(set(words))}")

word_freq = Counter(words)

# ── Top-N most frequent words ─────────────────────────────────────────────────
TOP_N = 30
print(f"\n{'-'*45}")
print(f"Top {TOP_N} most frequent words:")
print(f"{'-'*45}")
print(f"  {'Word':<30} {'Count':>6}")
print(f"  {'-'*30} {'-'*6}")
for word, count in word_freq.most_common(TOP_N):
    print(f"  {word:<30} {count:>6}")

# ── Rare words (appear only once) ─────────────────────────────────────────────
hapax = [w for w, c in word_freq.items() if c == 1]
print(f"\n{'-'*45}")
print(f"Hapax legomena (words appearing exactly once): {len(hapax)}")
print("  Sample (first 20):")
for w in sorted(hapax)[:20]:
    print(f"    {w}")

# ── Words appearing ≤ 3 times ────────────────────────────────────────────────
rare = [(w, c) for w, c in word_freq.items() if c <= 3]
rare.sort(key=lambda x: x[1])
print(f"\nRare words (freq ≤ 3): {len(rare)}")

# ── Symbol / punctuation analysis ────────────────────────────────────────────
SYMBOLS_OF_INTEREST = {
    ",":  "Comma",
    ".":  "Period",
    ":":  "Colon",
    "։":  "Armenian full stop (։)",
    "՝":  "Armenian but",
    "«":  "Left guillemet",
    "»":  "Right guillemet",
    "—":  "Em dash",
    "-":  "Hyphen",
    "՛":  "Armenian exclamation",
    "՞":  "Armenian question mark",
    "(":  "Left paren",
    ")":  "Right paren",
    "[":  "Left bracket",
    "]":  "Right bracket",
    ";":  "Semicolon",
    "՜":  "Armenian exitement mark",
}

print(f"\n{'-'*45}")
print("Symbol / punctuation counts:")
print(f"{'-'*45}")
print(f"  {'Symbol':<6} {'Name':<30} {'Count':>6}")
print(f"  {'-'*6} {'-'*30} {'-'*6}")
for sym, name in SYMBOLS_OF_INTEREST.items():
    count = raw_text.count(sym)
    print(f"  {sym!r:<6} {name:<30} {count:>6}")

# ── Full character-frequency breakdown (non-alphanumeric) ─────────────────────
non_alpha = [ch for ch in raw_text if not ch.isalpha() and not ch.isdigit() and not ch.isspace()]
char_freq = Counter(non_alpha)
print(f"\n{'-'*45}")
print("All non-alpha/digit/space characters (ranked):")
print(f"{'-'*45}")
print(f"  {'Char':<6} {'Unicode':<12} {'Count':>6}")
print(f"  {'-'*6} {'-'*12} {'-'*6}")
for ch, cnt in char_freq.most_common():
    print(f"  {ch!r:<6} U+{ord(ch):04X}      {cnt:>6}")

# ── Word-length distribution ──────────────────────────────────────────────────
from collections import defaultdict

length_dist: dict[int, int] = defaultdict(int)
for w in words:
    length_dist[len(w)] += 1

print(f"\n{'-'*45}")
print("Word-length distribution:")
print(f"{'-'*45}")
print(f"  {'Length':>6}  {'Count':>6}  Bar")
print(f"  {'-'*6}  {'-'*6}  {'-'*30}")
for length in sorted(length_dist):
    count = length_dist[length]
    bar = "#" * min(count // 5 + 1, 40)
    print(f"  {length:>6}  {count:>6}  {bar}")

# ── Average word length ───────────────────────────────────────────────────────
avg_len = sum(len(w) for w in words) / len(words) if words else 0
print(f"\nAverage word length: {avg_len:.2f} characters")

# ── Vocabulary richness (Type-Token Ratio) ────────────────────────────────────
ttr = len(set(words)) / len(words) if words else 0
print(f"Type-Token Ratio (TTR): {ttr:.4f}  (1.0 = all unique)")

# ── Digit / number tokens ─────────────────────────────────────────────────────
digit_tokens = [t for t in words_raw if re.fullmatch(r"\d+", t)]
print(f"\nNumeric tokens: {len(digit_tokens)}")
num_freq = Counter(digit_tokens)
print("  Top numeric tokens:", num_freq.most_common(10))

# =============================================================================
# VISUALIZATIONS
# =============================================================================
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ARMENIAN_FONT = "Sylfaen"   # Windows font with full Armenian Unicode support
OUT_DIR = "visualizations"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Match the style of visualizations.py exactly ─────────────────────────────
BG      = "#F8F8F8"
C_BLUE  = "#4C72B0"   # single chars / primary bars
C_ORA   = "#DD8452"   # subword / mean line
C_GRN   = "#55A868"   # full words / long words

matplotlib.rcParams["font.family"]          = "DejaVu Sans"
matplotlib.rcParams["axes.spines.top"]      = False
matplotlib.rcParams["axes.spines.right"]    = False
matplotlib.rcParams["figure.facecolor"]     = BG
matplotlib.rcParams["axes.facecolor"]       = BG


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


# ── Plot 1: Top-25 most frequent words (horizontal bar) ──────────────────────
top_words, top_counts = zip(*word_freq.most_common(25))

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor(BG)

y_pos = range(len(top_words))
bars = ax.barh(
    list(y_pos), list(top_counts),
    color=C_BLUE, edgecolor="white", linewidth=0.8, height=0.72,
)
ax.invert_yaxis()
ax.set_yticks(list(y_pos))
ax.set_yticklabels(list(top_words), fontname=ARMENIAN_FONT, fontsize=11)
ax.set_xlabel("Frequency in corpus", fontsize=12)
ax.set_title("Top 25 Most Frequent Words", fontsize=15, fontweight="bold", pad=14)
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

for bar, val in zip(bars, top_counts):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", ha="left", fontsize=9, color="#555555")

save(fig, "top_words.png")

# ── Plot 2: Word-length distribution (bar, coloured by category) ──────────────
lengths = sorted(length_dist)
counts  = [length_dist[l] for l in lengths]

def length_color(l):
    if l == 1:   return C_BLUE
    if l <= 4:   return C_ORA
    return C_GRN

bar_colors = [length_color(l) for l in lengths]

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG)

ax.bar(lengths, counts, color=bar_colors, edgecolor="white", linewidth=0.8, width=0.75)
ax.axvline(avg_len, color="#333333", linestyle="--", linewidth=1.4,
           label=f"Mean = {avg_len:.2f} chars")

# Vertical category dividers
ax.axvline(1.5, color="#aaaaaa", linestyle=":", linewidth=1, alpha=0.7)
ax.axvline(4.5, color="#aaaaaa", linestyle=":", linewidth=1, alpha=0.7)
top_y = max(counts) * 0.95
ax.text(1,   top_y, "single\nchars",        ha="center", va="top", fontsize=9, color=C_BLUE)
ax.text(3,   top_y, "subword\nfragments",   ha="center", va="top", fontsize=9, color=C_ORA)
ax.text(10,  top_y, "full words",           ha="center", va="top", fontsize=9, color=C_GRN)

ax.set_xlabel("Word length (characters)", fontsize=12)
ax.set_ylabel("Token count", fontsize=12)
ax.set_title("Word-Length Distribution", fontsize=15, fontweight="bold", pad=14)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.legend(fontsize=10, framealpha=0.85)

save(fig, "word_length_distribution.png")

print(f"\nPlots saved to '{OUT_DIR}/'.")
