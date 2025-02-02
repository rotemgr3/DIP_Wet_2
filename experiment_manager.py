import matplotlib.pyplot as plt
import os
import torch
import json


class ExperimentManager:
    def __init__(self, config):
        self.config = config
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        print("Experiment output directory:", self.output_dir)

    def save_config(self):
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=4)
        print("Saved config to", config_path)

    def save_model(self, model, epoch):
        model_path = os.path.join(
            self.output_dir, "models", f"model_epoch_{epoch}.pth")
        os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Saved model weights to {model_path}")

    def save_loss_plot(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(self.output_dir, "loss_curve.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved loss plot to {plot_path}")

    def save_psnr_plot(self, train_psnr, val_psnr):
        plt.figure(figsize=(10, 6))
        plt.plot(train_psnr, label='Train PSNR')
        plt.plot(val_psnr, label='Validation PSNR')
        plt.xlabel("Epoch")
        plt.ylabel("PSNR (dB)")
        plt.title("Training and Validation PSNR")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(self.output_dir, "psnr_curve.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved PSNR plot to {plot_path}")

    def save_examples(self, dataloader, model, mode="train", num_examples=5):
        model.eval()
        example_dir = os.path.join(self.output_dir, f"{mode}_examples")
        os.makedirs(example_dir, exist_ok=True)
        with torch.no_grad():
            for i, (noisy, clean) in enumerate(dataloader):
                if i >= num_examples:
                    break
                noisy = noisy.to(self.device, dtype=torch.float32) / 255.0
                clean = clean.to(self.device, dtype=torch.float32) / 255.0
                outputs = model(noisy)
                noisy_np = noisy.cpu().numpy()[0].transpose(1, 2, 0)
                clean_np = clean.cpu().numpy()[0].transpose(1, 2, 0)
                output_np = outputs.cpu().numpy()[0].transpose(1, 2, 0)

                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(noisy_np)
                plt.title("Noisy Input")
                plt.subplot(1, 3, 2)
                plt.imshow(clean_np)
                plt.title("Clean Target")
                plt.subplot(1, 3, 3)
                plt.imshow(output_np)
                plt.title("Denoised Output")
                plt.tight_layout()
                plt.savefig(os.path.join(example_dir, f"example_{i}.png"))
                plt.close()
        print(f"Saved {mode} examples to {example_dir}")

    def save_metrics(self, metrics_dict, metrics_path):
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        print("Saved metrics to", metrics_path)
