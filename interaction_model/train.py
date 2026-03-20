import torch
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2


class ParamToResidualCNN(nn.Module):
    def __init__(self, param_dim=4, base_channels=128):
        super().__init__()

        # Map params to latent vector
        self.fc = nn.Sequential(
            nn.Linear(param_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, base_channels * 16 * 16),
        )
        
        # Decode to image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, params):
        x = self.fc(params)
        x = x.view(-1, 128, 16, 16)
        x = self.decoder(x)
        return x


from dataset import SyntheticInteractionDataset, TorchDatasetWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load base image
base_image = cv2.imread("../param_none.png", cv2.IMREAD_COLOR)
base_image = cv2.resize(base_image, (256, 256))

# Dataset
train_dataset = SyntheticInteractionDataset("../synthetic_dataset", "train")
train_dataset = TorchDatasetWrapper(train_dataset, base_image)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)

# Model
model = ParamToResidualCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.L1Loss()


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

def train():
    epochs = 30

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for params, residual in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            params   = params.to(device)
            residual = residual.to(device)
            optimizer.zero_grad()

            pred_residual = model(params)
            loss = criterion(pred_residual, residual)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:.6f}  lr = {current_lr:.2e}")

    torch.save(model.state_dict(), "param_to_residual_cnn.pth")


if __name__ == "__main__":
    train()

    model = ParamToResidualCNN().to(device)
    model.load_state_dict(torch.load("param_to_residual_cnn.pth", map_location=device))
    model.eval()
    
    val_dataset = SyntheticInteractionDataset("../synthetic_dataset", "val")
    val_dataset = TorchDatasetWrapper(val_dataset, base_image)
    
    import matplotlib.pyplot as plt

    num_samples = 10
    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 6))

    with torch.no_grad():
        for i in range(num_samples):
            params, residual_gt = val_dataset[i]
            params = params.unsqueeze(0).to(device)

            pred_residual = model(params).cpu()[0]
            base = val_dataset.base_image

            pred_image = torch.clamp(base + pred_residual, 0, 1)
            gt_image   = torch.clamp(base + residual_gt,   0, 1)
            axes[0, i].imshow(gt_image.permute(1, 2, 0).numpy())
            axes[0, i].axis("off")
            axes[1, i].imshow(pred_image.permute(1, 2, 0).numpy())
            axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Ground Truth", fontsize=12)
    axes[1, 0].set_ylabel("Predicted", fontsize=12)

    plt.tight_layout()
    plt.show()