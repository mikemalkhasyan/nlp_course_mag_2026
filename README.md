# NLP Course

Homework and project assignments for the **Natural Language Processing** course at the **National Polytechnic University of Armenia (NPUA)**, Master's program, second semester.

Each directory contains a standalone project with its own dependencies, instructions, and documentation.

---

## Projects

| # | Project                                   | Description                                                              |
|---|-------------------------------------------|--------------------------------------------------------------------------|
| 1 | [LDA Topic Modeling](lda_topic_modeling/) | Unsupervised topic discovery using Latent Dirichlet Allocation (Gensim)  |
| 2 | [SentencePiece BPE](sentencepiece_bpe/)   | BPE tokenizer trained on an Armenian corpus using SentencePiece          |

---

## Project Summaries

### 1. LDA Topic Modeling — [`lda_topic_modeling/`](lda_topic_modeling/)

An end-to-end pipeline for training, labeling, and running inference with an LDA topic model.

- Trains an LDA model (7 topics, 30 passes) on a text corpus
- Provides an interactive interface for assigning human-readable labels to discovered topics
- Classifies new documents and returns per-topic probability scores
- Generates visualizations: word clouds, heatmaps, bar charts, and document-topic distributions

**Stack:** Python, Gensim, NLTK, Matplotlib, WordCloud

---

### 2. SentencePiece BPE Tokenizer — [`sentencepiece_bpe/`](sentencepiece_bpe/)

Trains a BPE tokenizer on a small Armenian corpus and analyses the resulting vocabulary.

- Trains a 300-token BPE model with full Armenian Unicode character coverage
- Encodes and decodes three test sentences with round-trip verification
- Analyses vocabulary structure: single characters, subword fragments, and full words
- Generates four plots: vocab composition donut, token frequency chart, length histogram, and sentence tokenization diagram

**Stack:** Python, SentencePiece, Matplotlib, NumPy

---

## Repository Structure

```text
nlp-course-npua/
├── lda_topic_modeling/     # Homework 1 — LDA Topic Modeling
│   ├── 1_training.py
│   ├── 2_labeling.py
│   ├── 3_inference.py
│   ├── visualizations.py
│   ├── models/
│   └── visualizations/
└── sentencepiece_bpe/      # Homework 2 — SentencePiece BPE Tokenizer
    ├── 1_training.py
    ├── 2_encoding_decoding.py
    ├── 3_vocabulary_analysis.py
    ├── visualizations.py
    ├── corpus.txt
    ├── models/
    └── visualizations/
```

---

## Requirements

Each project manages its own virtual environment and dependencies. Refer to the `README.md` inside each project directory for setup instructions.

General prerequisites:

- Python 3.8+
- `pip`

---
