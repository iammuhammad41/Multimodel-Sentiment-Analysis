# Multimodal Sentiment Analysis Pipeline

A PyTorch-based framework for multimodal representation learning and sentiment prediction. This repository provides:

* **Text Encoding** via pre-trained BERT
* **Audio & Video Encoding** via LSTM feature encoders
* **Low‑Level Reconstruction** for self‑supervised representation alignment
* **High‑Level Attraction** to measure feature similarity
* **Transformer‑Based Sentiment Predictor** to fuse modalities


## Repository Structure

```
.
├── data_loader.py               # Load & preprocess text, audio, video data
├── lstm_feature_encoder.py      # LSTMEncoder for temporal feature extraction
├── bert_text_encoder.py         # TextEncoder wraps BERT for embedding sentences
├── low_level_reconstruction.py  # Auto‑encoding + attraction modules
├── transformer_sentiment_predictor.py
│   └── TransformerSentimentPredictor  # Multimodal transformer classifier
├── main.py                      # Example script showing end‑to‑end usage
└── requirements.txt             # Python dependencies
```



## Features

* **Modular Encoders**

  * `TextEncoder`: BERT-based sentence embeddings
  * `LSTMEncoder`: learns temporal representations from audio/video
* **Self‑Supervised Reconstruction**

  * `LowLevelReconstruction`: MSE reconstruction loss
  * `HighLevelAttraction`: cosine similarity between “complete” & “incomplete” views
* **Multimodal Fusion**

  * Stack text, audio, and video features
  * Pass through a small Transformer encoder
  * Predict sentiment labels with a softmax classifier
* **Save & Load**

  * Each encoder supports `.save_model()` / `.load_model()`
  * Text features can be persisted via `torch.save`


## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/multimodal-sentiment-pipeline.git
   cd multimodal-sentiment-pipeline
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   *Requirements include:*

   * torch
   * transformers
   * librosa
   * numpy



## Usage

1. **Prepare your data**

   * Text files: plain `.txt`
   * Audio files: `.wav`
   * Video features: pre‑computed feature arrays (e.g. from a CNN)

2. **Edit `main.py`**
   Update file paths and hyperparameters under the **Initialize components** section.

3. **Run end‑to‑end script**:

   ```bash
   python main.py
   ```

   This will:

   * Load & preprocess each modality
   * Encode text/audio/video features
   * Predict sentiment labels
   * Compute reconstruction losses and feature similarities



## Module Details

### `data_loader.py`

* **`DataLoader`**

  * `load_text_data()`, `tokenize_text()`
  * `load_audio_data()`: extracts 40‑MFCC per file
  * `augment_audio_data()`, `normalize_audio_data()`
  * `get_multimodal_data()`: returns tokenized text, audio tensor, video tensor, and simple text‑length features

### `lstm_feature_encoder.py`

* **`LSTMEncoder`**

  * `forward(x)`: returns encoded feature vector
  * `extract_temporal_features(x)`: averages LSTM outputs over time

### `bert_text_encoder.py`

* **`TextEncoder`**

  * Wraps HuggingFace BERT
  * `encode_text(text_list)`: mean‑pool token embeddings
  * `extract_advanced_features()`: concatenates `[CLS]` and mean embeddings

### `low_level_reconstruction.py`

* **`LowLevelReconstruction`**: fully‑connected autoencoder + MSE loss
* **`HighLevelAttraction`**: maps incomplete view via FC, computes cosine similarity to complete view

### `transformer_sentiment_predictor.py`

* **`TransformerSentimentPredictor`**

  * Encodes text via BERT, audio/video via `LSTMEncoder`
  * Stacks modalities into a sequence of length 3
  * Processes with a 2‑layer Transformer encoder
  * Classifies into `num_classes` with softmax


## Customization

* **Hyperparameters** are defined in `main.py` for easy tuning:

  ```python
  # e.g.:
  audio_encoder = LSTMEncoder(input_size=40, hidden_size=128, num_layers=2)
  sentiment_predictor = TransformerSentimentPredictor(
      text_hidden_size=768,
      audio_hidden_size=128,
      video_hidden_size=128,
      transformer_hidden_size=256,
      num_heads=4,
      num_classes=3
  )
  ```
* **Training & Evaluation**

  * Extend `main.py` to include an explicit training loop
  * Compute task‑specific loss (e.g. cross‑entropy) and backpropagate
