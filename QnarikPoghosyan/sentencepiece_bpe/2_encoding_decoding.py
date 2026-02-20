"""
SentencePiece BPE Tokenizer — Part 2: Encoding and Decoding Script
===================================================================
Loads the trained BPE model and applies it to three Armenian test sentences.

Steps:
1. Load the trained model
2. For each sentence: encode to pieces, encode to IDs, decode back
3. Verify that decoded text matches the original exactly
"""

import sentencepiece as spm


MODEL_PATH = "models/hy_bpe.model"

TEST_SENTENCES = [
    "Հայաստանն ունի հարուստ պատմություն։",
    "Արհեստական բանականությունը արագ զարգանում է։",
    "Ծրագրավորումը կարևոր հմտություն է ապագայի համար։",
]


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
    print(f"✓ Loaded model from '{model_path}'  "
          f"(vocab size: {sp.get_piece_size()})\n")
    return sp


# STEP 2: ENCODE AND DECODE

def encode_decode(sp, sentence):
    """
    Encode a sentence to token pieces and IDs, then decode back.

    Args:
        sp:       Loaded SentencePieceProcessor
        sentence: Input text string

    Returns:
        tuple: (pieces, ids, decoded, match)
    """
    pieces  = sp.encode(sentence, out_type=str)
    ids     = sp.encode(sentence, out_type=int)
    decoded = sp.decode(ids)
    match   = decoded == sentence
    return pieces, ids, decoded, match


# STEP 3: DISPLAY RESULTS

def display_results(label, sentence, pieces, ids, decoded, match):
    """
    Print the full encoding/decoding report for a single sentence.

    Args:
        label:    Short identifier (e.g. 'S1')
        sentence: Original input text
        pieces:   List of token piece strings
        ids:      List of token IDs (integers)
        decoded:  Decoded string
        match:    Boolean — does decoded equal the original?
    """
    sep = "-" * 60
    print(sep)
    print(f"{label}: {sentence}")
    print(f"  Pieces  ({len(pieces):>3} tokens) : {pieces}")
    print(f"  IDs               : {ids}")
    print(f"  Decoded           : {decoded}")
    print(f"  Round-trip match  : {match}")


# MAIN EXECUTION

def main():
    print("\n" + "=" * 60)
    print("SENTENCEPIECE BPE — ENCODING & DECODING")
    print("=" * 60 + "\n")

    # Step 1: Load model
    sp = load_model(MODEL_PATH)

    # Step 2 & 3: Process each sentence
    for i, sentence in enumerate(TEST_SENTENCES, start=1):
        pieces, ids, decoded, match = encode_decode(sp, sentence)
        display_results(f"S{i}", sentence, pieces, ids, decoded, match)

    print("-" * 60)
    print("\n✓ Encoding/decoding complete!")


if __name__ == "__main__":
    main()
