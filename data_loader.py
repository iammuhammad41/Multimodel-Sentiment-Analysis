
import os
import numpy as np
import librosa
from transformers import BertTokenizer, BertModel
import torch

class DataLoader:
    def __init__(self, text_files, audio_files, video_features):
        self.text_files = text_files
        self.audio_files = audio_files
        self.video_features = video_features
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def load_text_data(self):
        text_data = []
        for file_path in self.text_files:
            with open(file_path, 'r') as file:
                text_data.append(file.read())
        return text_data

    def tokenize_text(self, text_data):
        return self.tokenizer(text_data, padding=True, truncation=True, return_tensors="pt")

    def load_audio_data(self):
        audio_features = []
        for file_path in self.audio_files:
            y, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            audio_features.append(np.mean(mfcc, axis=1))
        return torch.tensor(audio_features, dtype=torch.float)

    def load_video_features(self):
       
        return torch.tensor(self.video_features, dtype=torch.float)

    def augment_audio_data(self, audio_data):
        
        noise = torch.randn_like(audio_data) * 0.005
        augmented_audio = audio_data + noise
        return augmented_audio

    def normalize_audio_data(self, audio_data):
        
        return (audio_data - audio_data.min()) / (audio_data.max() - audio_data.min()) * 2 - 1

    def extract_text_features(self, tokenized_text):
        
        return torch.tensor([len(tokens) for tokens in tokenized_text["input_ids"]], dtype=torch.float)

    def process_all_data(self):
        text_data = self.load_text_data()
        tokenized_text = self.tokenize_text(text_data)
        audio_data = self.load_audio_data()
        video_data = self.load_video_features()

        augmented_audio = self.augment_audio_data(audio_data)
        normalized_audio = self.normalize_audio_data(augmented_audio)
        text_features = self.extract_text_features(tokenized_text)

        return tokenized_text, normalized_audio, video_data, text_features

    def get_multimodal_data(self):
        tokenized_text, normalized_audio, video_data, text_features = self.process_all_data()
        return {
            "tokenized_text": tokenized_text,
            "audio_data": normalized_audio,
            "video_data": video_data,
            "text_features": text_features
        }


if __name__ == "__main__":
    text_files = ["sample_text_1.txt", "sample_text_2.txt"]
    audio_files = ["sample_audio_1.wav", "sample_audio_2.wav"]
    video_features = [np.random.rand(10, 256), np.random.rand(10, 256)]

    loader = DataLoader(text_files, audio_files, video_features)
    data = loader.get_multimodal_data()
    print(data)
