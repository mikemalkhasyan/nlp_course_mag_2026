"""
SentencePiece BPE Tokenizer — Part 3: Vocabulary Analysis Script
=================================================================
Analyses the structure of the trained BPE vocabulary and measures
token frequency across the full corpus.

Steps:
1. Load the trained model
2. Categorise vocabulary entries by token length
3. Count token frequencies across the entire corpus
4. Report the 10 most frequent pieces

Vocabulary length categories (▁ prefix stripped before measuring):
    Single characters : length == 1
    Subword fragments : length 2–4
    Full words        : length 5+
"""

from collections import Counter
import sentencepiece as spm


MODEL_PATH  = "models/hy_bpe.model"
CORPUS_PATH = "corpus.txt"


# STEP 1: LOAD MODEL

def load_model(model_path):
    """
    Load a trained SentencePiece model from disk.

    Args:
        model_path: Path to the .model file

    Returns:
        Loaded SentencePieceProcessor
    """
    import os
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "Please run '1_training.py' first."
        )
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


# STEP 2: CATEGORISE VOCABULARY

def categorise_vocabulary(sp):
    """
    Categorise every vocabulary entry by its surface length.

    The leading ▁ (word-boundary marker) is stripped before measuring so
    that a piece like '▁կ' is counted as a single character, not two.

    Args:
        sp: Loaded SentencePieceProcessor

    Returns:
        dict with keys 'single_chars', 'subword_frags', 'full_words'
              and lists of the corresponding pieces
    """
    single_chars = []
    subword_frags = []
    full_words    = []

    for i in range(sp.get_piece_size()):
        piece  = sp.id_to_piece(i)
        length = len(piece.lstrip("▁"))

        if length == 1:
            single_chars.append(piece)
        elif 2 <= length <= 4:
            subword_frags.append(piece)
        else:
            full_words.append(piece)

    return {
        "single_chars": single_chars,
        "subword_frags": subword_frags,
        "full_words":    full_words,
    }


# STEP 3: COUNT CORPUS TOKEN FREQUENCIES

def corpus_token_frequencies(sp, corpus_path):
    """
    Encode every line of the corpus and count how often each token piece
    appears in total.

    Args:
        sp:          Loaded SentencePieceProcessor
        corpus_path: Path to the plain-text corpus file

    Returns:
        collections.Counter  {piece: count}
    """
    import os
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found at '{corpus_path}'.")

    counts = Counter()
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                counts.update(sp.encode(line, out_type=str))
    return counts


# STEP 4: DISPLAY ANALYSIS

def display_analysis(categories, token_counts, top_n=10):
    """
    Print the vocabulary breakdown and top-N token frequency table.

    Args:
        categories:   Result of categorise_vocabulary()
        token_counts: Result of corpus_token_frequencies()
        top_n:        How many top tokens to list (default 10)
    """
    total = sum(len(v) for v in categories.values())

    print("Vocabulary breakdown:")
    print(f"  {'Single characters (len = 1)':<35} {len(categories['single_chars']):>4}"
          f"  ({100 * len(categories['single_chars']) / total:.1f}%)")
    print(f"  {'Subword fragments  (len 2–4)':<35} {len(categories['subword_frags']):>4}"
          f"  ({100 * len(categories['subword_frags']) / total:.1f}%)")
    print(f"  {'Full words         (len 5+)':<35} {len(categories['full_words']):>4}"
          f"  ({100 * len(categories['full_words']) / total:.1f}%)")
    print(f"  {'Total':<35} {total:>4}")

    print(f"\nTop {top_n} most frequent token pieces in the corpus:")
    print(f"  {'Rank':>4}  {'Piece':<22}  {'Count':>6}")
    print("  " + "-" * 38)
    for rank, (piece, count) in enumerate(token_counts.most_common(top_n), start=1):
        print(f"  {rank:>4}  {piece:<22}  {count:>6}")


# MAIN EXECUTION

def main():
    import os

    print("\n" + "=" * 60)
    print("SENTENCEPIECE BPE — VOCABULARY ANALYSIS")
    print("=" * 60 + "\n")

    # Step 1: Load model
    sp = load_model(MODEL_PATH)
    print(f"✓ Loaded model  (vocab size: {sp.get_piece_size()})\n")

    # Step 2: Categorise vocabulary
    categories = categorise_vocabulary(sp)

    # Step 3: Count corpus frequencies
    token_counts = corpus_token_frequencies(sp, CORPUS_PATH)

    # Step 4: Display
    display_analysis(categories, token_counts, top_n=10)

    print("\n✓ Vocabulary analysis complete!")


if __name__ == "__main__":
    main()
