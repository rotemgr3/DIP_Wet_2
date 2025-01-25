import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from experiment_manager import ExperimentManager
from models import get_model
from dataset import SIDD_Dataset
from utils import compute_psnr


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    psnrs = []
    for noisy, clean in tqdm(dataloader, desc="Training"):
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        outputs = model(noisy)
        loss = criterion(outputs, clean)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # Compute PSNR for batch
        for i in range(noisy.size(0)):
            psnrs.append(compute_psnr(clean[i].detach().cpu().numpy(), outputs[i].detach().cpu().numpy()))
    avg_psnr = sum(psnrs) / len(psnrs)
    return running_loss / len(dataloader), avg_psnr

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    psnrs = []
    with torch.no_grad():
        for noisy, clean in tqdm(dataloader, desc="Evaluating"):
            noisy, clean = noisy.to(device), clean.to(device)
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            running_loss += loss.item()
            for i in range(noisy.size(0)):
                psnrs.append(compute_psnr(clean[i].cpu().numpy(), outputs[i].cpu().numpy()))
    avg_psnr = sum(psnrs) / len(psnrs)
    return running_loss / len(dataloader), avg_psnr

def train_model(config):
    # Experiment setup
    manager = ExperimentManager(config)
    manager.save_config()

    # Load datasets
    train_dataset = SIDD_Dataset(config["dataset_dir"], crop_size=config["crop_size"], mode="train")
    val_dataset = SIDD_Dataset(config["dataset_dir"], crop_size=config["crop_size"], mode="val")
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))

    # Initialize model, loss, and optimizer
    model = get_model(config).to(manager.device)
    print(f'Model architecture:\n{model}')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Track training and validation losses
    train_losses = []
    val_losses = []
    train_psnr_list = []
    val_psnr_list = []

    # Early stopping parameters
    early_stopping_patience = 5
    no_improvement_epochs = 0
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(config["epochs"]):
        train_loss, train_psnr = train_one_epoch(model, train_dataloader, criterion, optimizer, manager.device)
        val_loss, val_psnr = evaluate_model(model, val_dataloader, criterion, manager.device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_psnr_list.append(train_psnr)
        val_psnr_list.append(val_psnr)

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Train PSNR = {train_psnr:.2f} dB, Val PSNR = {val_psnr:.2f} dB")

        # Save model only if validation loss has decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_epochs = 0
            manager.save_model(model, epoch + 1)
        else:
            no_improvement_epochs += 1

        # Check early stopping condition
        if no_improvement_epochs >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break
    
    print("Training complete.")

    # Save final outputs
    manager.save_loss_plot(train_losses, val_losses)
    manager.save_metrics({"train_psnr": train_psnr_list, "val_psnr": val_psnr_list}, os.path.join(config['output_dir'], "metrics.json"))
    manager.save_psnr_plot(train_psnr_list, val_psnr_list)
    manager.save_examples(train_dataloader, model, mode="train")
    manager.save_examples(val_dataloader, model, mode="val")

def test_model(config):
    # Experiment setup
    manager = ExperimentManager(config)

    # Load test dataset
    test_dataset = SIDD_Dataset(config["dataset_dir"], crop_size=config["crop_size"], mode="test")
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    print("Test dataset size:", len(test_dataset))

    # search last epoch model weights file "model_epoch_<epoch>.pth"
    model_path = os.path.join(config["output_dir"], "models")
    weights_files = [f for f in os.listdir(model_path) if f.endswith(".pth")]
    weights_files.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))
    
    latest_weights_path = os.path.join(model_path, weights_files[-1])
    model = get_model(config).to(manager.device)
    model.load_state_dict(torch.load(latest_weights_path))
    
    print(f"Loaded model weights from {latest_weights_path}")
    
    model.eval()

    # Evaluate model on test set
    criterion = nn.MSELoss()
    test_loss, test_psnr = evaluate_model(model, test_dataloader, criterion, manager.device)
    print(f"Test Loss = {test_loss:.4f}, Test PSNR = {test_psnr:.2f} dB")

    # Save test metrics
    manager.save_metrics({"test_loss": test_loss, "test_psnr": test_psnr}, 
                         os.path.join(config['output_dir'], "test_metrics.json"))
    manager.save_examples(test_dataloader, model, mode="test")