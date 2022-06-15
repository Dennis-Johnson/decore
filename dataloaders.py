import torch, torchvision

def get_dataloaders(batch_size_train=64, batch_size_test=1000):
    transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        './datasets/', 
        train = True,
        download = True,
        transform = transforms),
    batch_size = batch_size_train,
    shuffle = True
    )

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        './datasets/',
        train = False,
        download = True,
        transform = transforms),
    batch_size = batch_size_test, 
    shuffle = True
    )

    return train_loader, test_loader