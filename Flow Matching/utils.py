import matplotlib.pyplot as plt 
import numpy as np 
import torch 

def show_images(images, n_cols=4):
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy() 

    if images.ndim == 2:
        images = images[np.newaxis]
    elif images.ndim == 3 and images.shape[0] == 1:
        images = np.repeat(images, 3, axis=0)
        images = images[np.newaxis]
    elif images.ndim == 3 and images.shape[0] == 3:
        images = images[np.newaxis]

    n = images.shape[0]
    n_rows = int(np.ceil(n / n_cols)) # calculate number of rows

    processed_images = [] 
    for img in images :
        img = np.transpose(img, (1, 2, 0))
        img = (img + 1) / 2 
        img = np.clip(img, 0, 1)
        processed_images.append(img)

    fig, axes = plt.subplots(n_rows, n_cols, figsize = (n_cols * 3, n_rows * 3))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(-1, 1)

    for ax, img in zip(axes.flat, processed_images):
        ax.imshow(img)
        ax.axis('off')
    
    for ax in axes.flat[n:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
