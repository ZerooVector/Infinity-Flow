from model import BasicScoreNet
from ScoreMatching import ScoreMatching
from utils import show_images
import torch


def sample(batch_size = 12):
    model = BasicScoreNet()
    sm = ScoreMatching(model)
    save_path = "./checkpoints/epoch_400.pt"
    model.load_state_dict(torch.load(save_path))
    
    samples = sm.generate_image(batch_size=batch_size)
    return samples

if __name__ == "__main__":
    samples = sample()
    show_images(samples)

    