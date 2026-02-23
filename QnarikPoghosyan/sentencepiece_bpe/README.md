# SentencePiece BPE Tokenizer — Armenian Corpus

Train and analyze a Byte-Pair Encoding (BPE) tokenizer with [SentencePiece](https://github.com/google/sentencepiece) on an Armenian text corpus.

## Contents

| File | Description |
| --- | --- |
| `1_training.py` | Train the BPE model, inspect first/last 30 vocabulary entries |
| `2_encoding_decoding.py` | Encode 3 test sentences to pieces and IDs, decode and verify round-trip |
| `3_vocabulary_analysis.py` | Categorize vocab by length, print top-10 corpus token frequencies |
| `visualizations.py` | Generate 4 plots into `visualizations/` |
| `corpus.txt` | 93-sentence Armenian corpus (558 words) |

## Running Order

```bash
pip install sentencepiece matplotlib numpy

python 0_corpus_analysis.py
python 1_training.py           # trains models/hy_bpe.model
python 2_encoding_decoding.py
python 3_vocabulary_analysis.py
python visualizations.py       # saves plots to visualizations/
```

## Visualizations

![Sentence tokenizer](visualizations/sentence_tokenization.png)
![Top tokens](visualizations/token_frequencies.png)

## Project Structure

```text
sentencepiece_bpe/
├── 1_training.py
├── 2_encoding_decoding.py
├── 3_vocabulary_analysis.py
├── visualizations.py
├── corpus.txt
├── models/                  # created by 1_training.py
│   ├── hy_bpe.model
│   └── hy_bpe.vocab
└── visualizations/          # created by visualizations.py
```

## Requirements

- Python 3.8+
- `sentencepiece`
- `matplotlib`
- `numpy`
