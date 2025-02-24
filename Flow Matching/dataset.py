import torch 
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from config import Config

def get_mnist_dataloader():

    cfg = Config() 

    transform = transforms.Compose([
        transforms.Resize((cfg.image_x_size, cfg.image_y_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x : x * 2 - 1)
    ])

    train = datasets.MNIST(
        root = "./data",
        train=True ,
        download=True,
        transform=transform
    )

    # train_subset = torch.utils.data.Subset(train, indices=range(cfg.num_data))  # 取0~1999索引

    test = datasets.MNIST(
        root="./data",
        train=False,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(train, batch_size=cfg.batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test,batch_size=cfg.batch_size,shuffle=False)
    
    return train_loader, test_loader



def get_data_loaders():

    cfg = Config() 

    if cfg.dataset == "MNIST":
        return get_mnist_dataloader() 
    

