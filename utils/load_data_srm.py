import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms


class CombinedDataset(Dataset):
    def __init__(self, image_dir, timeseries_dir, transform=None):
        """
        Args:
            image_dir (str): Directory containing image files.
            timeseries_dir (str): Directory containing time series text files.
            transform (callable, optional): Optional transform to be applied on the images.
        """
        self.image_dir = image_dir
        self.timeseries_dir = timeseries_dir
        self.transform = transform

        # Get sorted list of image and timeseries file paths
        self.image_paths = sorted([
            os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        self.timeseries_paths = sorted([
            os.path.join(timeseries_dir, fname) for fname in os.listdir(timeseries_dir)
            if fname.lower().endswith('.csv')
        ])
        
        assert len(self.image_paths) == len(self.timeseries_paths), (
            f"Number of images ({len(self.image_paths)}) and time series files ({len(self.timeseries_paths)}) must match"
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and transform the image
        image = Image.open(self.image_paths[idx]).convert("1")
        if self.transform:
            image = self.transform(image)
        
        # Load the corresponding time series data from text file
        timeseries_path = self.timeseries_paths[idx]
        timeseries = np.loadtxt(timeseries_path, dtype=np.float32)
        timeseries = torch.tensor(timeseries, dtype=torch.float32)

        return image, timeseries


if __name__ == "__main__":
    data_transforms = transforms.Compose([
        #transforms.Resize((1250, 1250)),  # Resize images
        transforms.ToTensor(),          # Convert to tensor
    ])

    # Example usage
    image_dir = 'test_data/'
    timeseries_dir = 'test_data_y/'

    combined_dataset = CombinedDataset(image_dir, timeseries_dir, transform=data_transforms)
    dataloader = DataLoader(combined_dataset, batch_size=2, shuffle=True)

    # Iterate through the DataLoader
    for images, timeseries in dataloader:
        print(f"Images batch shape: {images.shape}")
        print(f"Timeseries batch shape: {timeseries.shape}")
