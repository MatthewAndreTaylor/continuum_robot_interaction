import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TorchDatasetWrapper(Dataset):
    def __init__(self, base_dataset, base_image):
        base_image = base_image.astype(np.float32) / 255.0
        base_image = np.transpose(base_image, (2, 0, 1))  # HWC -> CHW
        base_tensor = torch.tensor(base_image)

        # Precompute (params, residual) for every sample
        self.samples = []
        for params, image in base_dataset:
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image)
            params = torch.tensor(params)
            residual = image - base_tensor
            self.samples.append((params, residual))

        self.base_image = base_tensor  # keep for inference

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class SyntheticInteractionDataset(Dataset):
    def __init__(self, data_dir, split, kappa_range=(5.0, 10.0), phi_range=(-np.pi, np.pi)):
        self.data_dir = data_dir
        self.samples = []

        # normalization constants
        self.kappa_range = kappa_range
        self.phi_range = phi_range

        self.kappa_diff = kappa_range[1] - kappa_range[0]
        self.phi_diff = phi_range[1] - phi_range[0]

        split = split.lower()
        self.load_samples()

        l = len(self.samples)
        if split == "train":
            self.samples = self.samples[: int(0.8 * l)]
        else:
            self.samples = self.samples[int(0.8 * l):]

    def load_samples(self):
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".png"):
                params = self.parse_params_from_filename(filename)
                if params is not None:
                    image_path = os.path.join(self.data_dir, filename)
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    image = cv2.resize(image, (256, 256))
                    self.samples.append((params, image))

    def parse_params_from_filename(self, filename):
        try:
            parts = filename[:-4].split('_')
            k1 = float(parts[2])
            phi1 = float(parts[4])
            k2 = float(parts[6])
            phi2 = float(parts[8])
            return self.normalize_params((k1, phi1, k2, phi2))
        except (IndexError, ValueError):
            print(f"Warning: Could not parse parameters from filename '{filename}'")
            return None

    def normalize_params(self, params):
        k1, phi1, k2, phi2 = params
        k1   = (k1   - self.kappa_range[0]) / self.kappa_diff
        k2   = (k2   - self.kappa_range[0]) / self.kappa_diff
        phi1 = (phi1 - self.phi_range[0])   / self.phi_diff
        phi2 = (phi2 - self.phi_range[0])   / self.phi_diff
        return np.array([k1, phi1, k2, phi2], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == "__main__":
    dataset = SyntheticInteractionDataset(data_dir="../synthetic_dataset", split="train")

    from matplotlib import pyplot as plt

    num_samples = 5

    base_image = cv2.imread("../param_none.png", cv2.IMREAD_COLOR)
    base_image = cv2.resize(base_image, (256, 256))

    fig, axes = plt.subplots(1, num_samples, figsize=(25, 6))

    for i in range(num_samples):
        params, image = dataset[i]

        diff_image = cv2.absdiff(image, base_image)
        combined_image = np.vstack((image, diff_image))

        axes[i].imshow(combined_image)
        axes[i].set_title(f"{i}: {params}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()