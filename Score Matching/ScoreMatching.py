import torch 
import torch.nn as nn
from config import Config
from tqdm import tqdm
from torchdiffeq import odeint

class ScoreMatching():
    def __init__(self, score_model):
        self.score_model = score_model 
        self.score_model.train()
        self.cfg = Config()
        self.score_model.to(self.cfg.device)
        self.criterion = nn.MSELoss()
        self.noise_schedule = torch.linspace(self.cfg.min_beta, self.cfg.max_beta, self.cfg.time_steps, device=self.cfg.device)
        self.dt = 1.0 / self.cfg.time_steps
        self.beta_integral = torch.cumsum(self.noise_schedule * self.dt, dim = 0).to(device=self.cfg.device)
        # print(self.beta_integral)



    def sample_normal_prior(self, batch_size):
        normal_prior = torch.randn(batch_size, self.cfg.image_channels, self.cfg.image_x_size, self.cfg.image_y_size, device=self.cfg.device)
        return normal_prior
    
    def get_noised_image(self, xT):
        batch_size = xT.shape[0] 
        t = torch.rand(batch_size, 1, device=self.cfg.device) * (1 - 1e-6)
        steps = torch.floor(t.squeeze(1) * self.cfg.time_steps).to(torch.int32)
        t_expand = t[:, :, None, None]
        standard_normal_noise = torch.randn_like(xT, device=self.cfg.device)
        mu = xT * (torch.exp(-self.beta_integral[steps] * 0.5)[:, None, None, None])
        var = torch.ones_like(xT, device=self.cfg.device) * (1 - torch.exp(-self.beta_integral[steps]))[:, None, None, None]
        noised_images = mu + torch.sqrt(var) * standard_normal_noise
        score_dominator = (1 - torch.exp( - self.beta_integral[steps]))[:, None, None, None]
        score = - (noised_images - mu) / (score_dominator + 1e-6)
        return t, noised_images, score

    
    def compute_train_loss(self, xT): 
        t, noised_images, score = self.get_noised_image(xT)
        predicted_score = self.score_model(noised_images, t)
        loss = self.criterion(score, predicted_score)
        return loss 
    
    def generate_image(self, batch_size):
        x = self.sample_normal_prior(batch_size=batch_size)
        dt = 1.0 / self.cfg.time_steps
        self.score_model.eval()
        with torch.no_grad():
            for t in tqdm(range(self.cfg.time_steps)):
                t0 = t / self.cfg.time_steps
                reverse_t = torch.tensor(1 - t0).to(device=self.cfg.device) * (1 - 1e-6)
                expand_reverse_t = torch.ones(batch_size, 1, device=self.cfg.device) * reverse_t
                steps = torch.floor(reverse_t * self.cfg.time_steps).to(torch.int32)
                beta = self.noise_schedule[steps]
                drift = 0.5 * beta * x + beta * self.score_model(x, expand_reverse_t)
                noise = torch.randn_like(x)
                dt = torch.tensor(dt, device=self.cfg.device)
                x = x + drift * dt + 1 * torch.sqrt(beta) * noise * torch.sqrt(dt)
        self.score_model.train()
        return x
    