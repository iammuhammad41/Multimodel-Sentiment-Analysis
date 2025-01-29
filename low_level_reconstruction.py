
import torch
import torch.nn as nn

class LowLevelReconstruction(nn.Module):
    def __init__(self, input_size):
        super(LowLevelReconstruction, self).__init__()
        self.fc = nn.Linear(input_size, input_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        reconstructed = self.fc(x)
        reconstructed = self.activation(reconstructed)
        return reconstructed

    def calculate_loss(self, original, reconstructed):
        loss_fn = nn.MSELoss()
        return loss_fn(original, reconstructed)

class HighLevelAttraction(nn.Module):
    def __init__(self, input_size):
        super(HighLevelAttraction, self).__init__()
        self.fc = nn.Linear(input_size, input_size)
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, incomplete_view, complete_view):
        transformed_incomplete = self.fc(incomplete_view)
        similarity = self.cosine_similarity(transformed_incomplete, complete_view)
        return similarity


if __name__ == "__main__":
    input_size = 256
    reconstruction = LowLevelReconstruction(input_size)
    original_data = torch.randn(5, input_size)
    reconstructed_data = reconstruction(original_data)
    loss = reconstruction.calculate_loss(original_data, reconstructed_data)
    print("Reconstruction Loss:", loss)

    high_level_attraction = HighLevelAttraction(input_size)
    incomplete_view = torch.randn(5, input_size)
    complete_view = torch.randn(5, input_size)
    similarity = high_level_attraction(incomplete_view, complete_view)
    print("Similarity Score:", similarity)
