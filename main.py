
from data_loader import DataLoader
from bert_text_encoder import TextEncoder
from lstm_feature_encoder import LSTMEncoder
from transformer_sentiment_predictor import TransformerSentimentPredictor
from low_level_reconstruction import LowLevelReconstruction, HighLevelAttraction
import torch

# Initialize components
text_files = ["sample_text_1.txt", "sample_text_2.txt"]
audio_files = ["sample_audio_1.wav", "sample_audio_2.wav"]
video_features = [torch.rand(10, 256), torch.rand(10, 256)]
loader = DataLoader(text_files, audio_files, video_features)

# Load data
data = loader.get_multimodal_data()
tokenized_text, audio_data, video_data, text_features = data["tokenized_text"], data["audio_data"], data["video_data"], data["text_features"]

# Text encoding
text_encoder = TextEncoder()
encoded_text = text_encoder.encode_text(["This is a sample text."])

# LSTM encoding for audio and video
audio_encoder = LSTMEncoder(40, 128, 2)
audio_encoded = audio_encoder(audio_data.unsqueeze(0))

video_encoder = LSTMEncoder(256, 128, 2)
video_encoded = video_encoder(video_data.unsqueeze(0))

# Transformer sentiment prediction
sentiment_predictor = TransformerSentimentPredictor(768, 128, 128, 256, 4, 3)
sentiment_output = sentiment_predictor(tokenized_text, audio_data, video_data)
print("Sentiment Prediction:", sentiment_output)

# Low-level reconstruction and high-level attraction
reconstruction = LowLevelReconstruction(256)
reconstructed_data = reconstruction(video_data)
loss = reconstruction.calculate_loss(video_data, reconstructed_data)
print("Reconstruction Loss:", loss)

high_level_attraction = HighLevelAttraction(256)
similarity = high_level_attraction(reconstructed_data, video_data)
print("High-Level Feature Similarity:", similarity)
