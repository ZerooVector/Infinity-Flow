from dataset import get_data_loaders
from model import BasicFlowNet
from flowmatching import FlowMatching
from config import Config
import torch.optim as optim 
import os 
import torch 
from tqdm import tqdm 

def train():
    cfg = Config()
    loader, _ = get_data_loaders()
    model = BasicFlowNet().to(cfg.device)
    fm = FlowMatching(model)

    if not os.path.exists(cfg.model_save_path):
        os.makedirs(cfg.model_save_path)

    optimizer = optim.AdamW(model.parameters(), lr = cfg.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma)

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0 
        batches = 0

        for batch, (x,_) in tqdm(enumerate(loader)):
            x = x.to(cfg.device)

            optimizer.zero_grad() 
            loss = fm.compute_train_loss(x)
            loss.backward() 
            optimizer.step() 

            epoch_loss += loss.item() 
            batches += 1


        scheduler.step()
        print(f"Epoch: {epoch}  | Loss: {epoch_loss / batches}")
        if epoch % cfg.model_save_interval == 0:
            save_path = f"{cfg.model_save_path}/epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), save_path)

    return 

if __name__ == "__main__":
    train()



