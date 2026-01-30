from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnnist_loaders(data_dir, batch_size=64):
    # Obtain train and test data
    train_data = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_data = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    # Make dataloader
    train_dataloader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
    )
    
    test_dataloader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=True, 
    )

    return train_dataloader, test_dataloader