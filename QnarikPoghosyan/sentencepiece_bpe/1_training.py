"""
SentencePiece BPE Tokenizer — Part 1: Training Script
======================================================
Trains a Byte-Pair Encoding model on an Armenian corpus using SentencePiece.

Steps:
1. Train the BPE model on corpus.txt
2. Load and inspect the resulting vocabulary
3. Print first and last 30 vocabulary entries
"""

import os
import sentencepiece as spm


CORPUS_PATH  = "corpus.txt"
MODEL_PREFIX = "models/hy_bpe"
VOCAB_SIZE   = 600


# STEP 1: TRAIN MODEL

def train_model():
    """
    Train a SentencePiece BPE model on the Armenian corpus.

    Parameters:
        input             - path to the plain-text training corpus
        model_prefix      - prefix for output files (.model and .vocab)
        vocab_size        - total number of tokens in the vocabulary
        model_type        - 'bpe' for Byte-Pair Encoding
        character_coverage - 1.0 ensures every Armenian Unicode character is covered
    """
    os.makedirs("models", exist_ok=True)

    print("Training SentencePiece BPE model...")
    spm.SentencePieceTrainer.train(
        input=CORPUS_PATH,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
    )
    print(f"✓ Model saved to '{MODEL_PREFIX}.model' and '{MODEL_PREFIX}.vocab'")


# STEP 2: INSPECT VOCABULARY

def inspect_vocabulary(sp):
    """
    Print vocabulary statistics and sample entries.

    Vocabulary entries fall into three broad categories:
        - Special control tokens  (<unk>, <s>, </s>)
        - Single Armenian characters (fallback coverage)
        - Subword fragments and full words learned by BPE

    Args:
        sp: Loaded SentencePieceProcessor
    """
    vocab_size = sp.get_piece_size()
    print(f"\nTotal vocabulary size: {vocab_size}")

    vocab = [(sp.id_to_piece(i), i) for i in range(vocab_size)]

    print("\nFirst 30 vocabulary entries:")
    print(f"  {'ID':>4}  {'Piece'}")
    print("  " + "-" * 30)
    for piece, idx in vocab[:30]:
        print(f"  {idx:>4}  {piece}")

    print("\nLast 30 vocabulary entries:")
    print(f"  {'ID':>4}  {'Piece'}")
    print("  " + "-" * 30)
    for piece, idx in vocab[-30:]:
        print(f"  {idx:>4}  {piece}")


# MAIN EXECUTION

def main():
    print("\n" + "=" * 60)
    print("SENTENCEPIECE BPE — TRAINING")
    print("=" * 60)

    # Step 1: Train
    train_model()

    # Step 2: Load and inspect
    sp = spm.SentencePieceProcessor()
    sp.load(f"{MODEL_PREFIX}.model")
    inspect_vocabulary(sp)

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
