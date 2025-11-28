# This file implements script to train a language model using PyTorch's Transformer modules. 


import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from model.model import LanguageModel

@hydra.main(config_path="../../configs", config_name="train_config", version_base=None)
def main(cfg: DictConfig):
    print("Starting training with configuration:")
    print("Model configuration:", cfg.model)
    print("Training configuration:", cfg.training)


    # Initialize model, optimizer, loss function
    model = LanguageModel(**cfg.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    loss_func = torch.nn.CrossEntropyLoss() 

    # DataLoader setup
    # Load the dataset and create train/val splits based on cfg.training.dataset_path
    dataset = PrepareDataset(cfg.training.dataset_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size)  
    
    

if __name__ == "__main__":
    main()