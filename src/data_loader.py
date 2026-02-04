from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_mnist_loaders(data_dir, batch_size=64, valex_size=0):
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

    # Get full train data
    train_dataloader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
    )

    # Get validation examples
    train_size = len(train_data) - valex_size
    
    train_data, valex_data = random_split(train_data, [train_size, valex_size])

    # Make dataloader
    valex_dataloader = DataLoader(
        valex_data, 
        batch_size=batch_size, 
        shuffle=True, 
    )

    trainex_dataloader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
    )
    
    test_dataloader = DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=True, 
    )

    

    return valex_dataloader, trainex_dataloader, train_dataloader, test_dataloader