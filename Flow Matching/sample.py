from model import BasicFlowNet
from flowmatching import FlowMatching
from utils import show_images
import torch


def sample(batch_size = 12):
    model = BasicFlowNet()
    fm = FlowMatching(model)
    save_path = "./checkpoints/epoch_86.pt"
    model.load_state_dict(torch.load(save_path))
    
    samples = fm.generate_image(batch_size=batch_size)
    return samples

if __name__ == "__main__":
    samples = sample()
    show_images(samples)

    