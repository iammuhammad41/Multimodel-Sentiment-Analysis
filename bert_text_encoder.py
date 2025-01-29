
import torch
from transformers import BertModel, BertTokenizer
import torch.nn as nn

class TextEncoder:
    def __init__(self):
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def encode_text(self, text_list):
        tokenized_text = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = self.bert_model(**tokenized_text)
        return output.last_hidden_state.mean(dim=1)

    def extract_advanced_features(self, tokenized_text):
        with torch.no_grad():
            output = self.bert_model(**tokenized_text)
            cls_embedding = output.last_hidden_state[:, 0, :]
            all_layers_mean = output.last_hidden_state.mean(dim=1)
            concatenated_features = torch.cat((cls_embedding, all_layers_mean), dim=1)
        return concatenated_features

    def save_features(self, features, output_path="text_features.pt"):
        torch.save(features, output_path)

    def load_features(self, path="text_features.pt"):
        return torch.load(path)

if __name__ == "__main__":
    encoder = TextEncoder()
    text_data = ["This is a test sentence.", "Here is another one."]
    encoded_text = encoder.encode_text(text_data)
    advanced_features = encoder.extract_advanced_features(encoder.tokenizer(text_data, return_tensors="pt"))
    print("Encoded Text:", encoded_text)
    print("Advanced Features:", advanced_features)
    encoder.save_features(advanced_features)
    loaded_features = encoder.load_features()
    print("Loaded Features:", loaded_features)
