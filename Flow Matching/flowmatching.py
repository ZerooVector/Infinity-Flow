import torch 
import torch.nn as nn
from config import Config
from tqdm import tqdm

class FlowMatching():
    def __init__(self, flow_model):
        self.flow_model = flow_model 
        self.flow_model.train()
        self.cfg = Config()
        self.flow_model.to(self.cfg.device)
        self.criterion = nn.MSELoss()

    def sample_normal_prior(self, batch_size):
        normal_prior = torch.randn(batch_size, self.cfg.image_channels, self.cfg.image_x_size, self.cfg.image_y_size, device=self.cfg.device)
        return normal_prior
    
    def compute_train_loss(self, xT):  # train conditional flow mathching given terminal points xT
        batch_size = xT.shape[0] 
        # print(xT.shape)
        t = torch.rand(batch_size, 1, device=self.cfg.device) 
        t_expand = t[:, :, None, None]
        normal_noise = torch.randn_like(xT)
        # print(normal_noise.shape)
        # shape of xT : [batchsize, c, x, y],  shape of t : [batchsize, 1]
        xt = t_expand * xT + (1 - t_expand) * normal_noise # linear interpolation

        target_vector_field = xT - normal_noise
        predicted_vector_field = self.flow_model(xt, t)
        loss = self.criterion(target_vector_field, predicted_vector_field)
        return loss 
    
    def generate_image(self, batch_size):
        x = self.sample_normal_prior(batch_size=batch_size)
        dt = 1.0 / self.cfg.time_steps
        self.flow_model.eval() 

        with torch.no_grad():
            for t in tqdm(range(self.cfg.time_steps)):
                vt = self.flow_model(x, torch.ones(batch_size, 1, device=self.cfg.device) * t / self.cfg.time_steps)
                x = x + vt * dt 
        
        self.flow_model.train()
        return x
    