from dataset import get_data_loaders
from model import BasicScoreNet, BasicScoreTransformer
from ScoreMatching import ScoreMatching
from config import Config
import torch.optim as optim 
import os 
import torch 
from tqdm import tqdm 

def train():
    cfg = Config()
    loader, _ = get_data_loaders()
    model = BasicScoreNet().to(cfg.device)

    # 如果需要加载预训练模型，可以取消下面两行的注释
    # save_path = "./checkpoints/epoch_76.pt"
    # model.load_state_dict(torch.load(save_path))
    
    sm = ScoreMatching(model)

    if not os.path.exists(cfg.model_save_path):
        os.makedirs(cfg.model_save_path)

    # 定义优化器，并使用 StepLR 实现学习率衰减
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma)
    # 注：这里定义的 StepLR 每隔 cfg.scheduler_step_size 个 epoch 将学习率乘以 gamma

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0 
        batches = 0

        for batch, (x, _) in tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{cfg.epochs}"):
            x = x.to(cfg.device)
            optimizer.zero_grad() 
            loss = sm.compute_train_loss(x)
            loss.backward() 
            optimizer.step() 

            epoch_loss += loss.item() 
            batches += 1

        # 每个 epoch 后更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch: {epoch+1} | Loss: {epoch_loss / batches:.4f} | LR: {current_lr:.6f}")

        if (epoch + 1) % cfg.model_save_interval == 0:
            save_path = f"{cfg.model_save_path}/epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), save_path)

    return

if __name__ == "__main__":
    train()



