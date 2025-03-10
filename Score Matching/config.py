import torch 

class Config:

    dataset = "MNIST"
    num_data = 20000000000
    image_x_size = 28 
    image_y_size = 28 
    image_channels = 1

    batch_size = 128 
    lr = 4e-4
    epochs = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_save_path = "./checkpoints"
    model_save_interval = 5
    scheduler_step_size = 5 
    scheduler_gamma = 0.98

    time_steps = 1250
    min_beta = 0.1
    max_beta = 1
     
