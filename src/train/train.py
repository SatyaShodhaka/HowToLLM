# This file implements script to train a language model using PyTorch's Transformer modules. 


import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from model.model import LanguageModel
from utils.data import LMDataset
import mlflow
import mlflow.pytorch

@hydra.main(config_path="../../configs", config_name="train_config", version_base=None)
def main(cfg: DictConfig):
    print("Starting training with configuration:")
    print("Model configuration:", cfg.model)
    print("Training configuration:", cfg.training)

    # Start MLflow run
    mlflow.set_experiment(cfg.training.get("mlflow_experiment", "HowToLLM"))
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(dict(cfg.model))
        mlflow.log_params(dict(cfg.training))

    # Initialize model, optimizer, loss function
    model = LanguageModel(**cfg.model)
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay) #We can also add learning rate schedulers here
    loss_func = torch.nn.CrossEntropyLoss() 

    # DataLoader setup
    # Use LMDataset to create tokenized, fixed-length samples for language modeling
    dataset = LMDataset(
        file_path=cfg.dataset.data_path,
        tokenizer_dir=cfg.dataset.tokenizer_path,
        context_length=cfg.model.context_length  # Make sure this matches your model config
    )
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # No need for a custom collate_fn since all samples are the same length
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size)
    

    # Training Loop
    for epoch in range(cfg.training.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = batch.to(device).transpose(0, 1)  # Now (seq_len, batch_size)
            optimizer.zero_grad()
            outputs = model(inputs[:-1, :])  # Predict next token
            loss = loss_func(outputs.view(-1, outputs.size(-1)), inputs[1:, :].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{cfg.training.epochs}, Training Loss: {avg_loss:.4f}")
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

        # Log learning rate and weight decay for the first param group (most common case)
        current_lr = optimizer.param_groups[0]['lr']
        current_wd = optimizer.param_groups[0].get('weight_decay', None)
        mlflow.log_metric("learning_rate", current_lr, step=epoch)
        if current_wd is not None:
            mlflow.log_metric("weight_decay", current_wd, step=epoch)

        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch.to(device)
                outputs = model(inputs[:-1, :])
                loss = loss_func(outputs.view(-1, outputs.size(-1)), inputs[1:, :].reshape(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{cfg.training.epochs}, Validation Loss: {avg_val_loss:.4f}")
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

    # Save the trained model: Can be also implemented to save best model based on validation loss
    torch.save(model.state_dict(), cfg.training.model_save_path)
    print(f"Model saved to {cfg.training.model_save_path}")
    mlflow.pytorch.log_model(model, cfg.training.model_save_path)
    print("Training complete.")

if __name__ == "__main__":
    main()