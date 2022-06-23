import torch, torchvision

def get_dataloaders(batch_size_train=64, batch_size_test=1000):
    transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(
        './datasets/', 
        train = True,
        download = True,
        transform = transforms),
    batch_size = batch_size_train,
    shuffle = True
    )

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(
        './datasets/',
        train = False,
        download = True,
        transform = transforms),
    batch_size = batch_size_test, 
    shuffle = True
    )

    return train_loader, test_loader