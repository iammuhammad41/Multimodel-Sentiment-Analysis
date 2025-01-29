
import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x)
        hidden = hidden[-1]
        encoded_features = self.fc(hidden)
        return encoded_features

    def extract_temporal_features(self, x):
        # Extract temporal information from LSTM output
        lstm_out, _ = self.lstm(x)
        temporal_features = lstm_out.mean(dim=1)
        return temporal_features

    def save_model(self, path="lstm_encoder.pth"):
        torch.save(self.state_dict(), path)

    def load_model(self, path="lstm_encoder.pth"):
        self.load_state_dict(torch.load(path))


if __name__ == "__main__":
    input_size = 40  
    hidden_size = 128
    num_layers = 2

    encoder = LSTMEncoder(input_size, hidden_size, num_layers)
    sample_data = torch.randn(10, 5, input_size)
    encoded_output = encoder(sample_data)
    temporal_features = encoder.extract_temporal_features(sample_data)
    print("Encoded Output:", encoded_output)
    print("Temporal Features:", temporal_features)
    encoder.save_model()
    encoder.load_model()
