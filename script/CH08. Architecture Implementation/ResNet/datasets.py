import torchvision
from torch.utils.data import DataLoader

# Data loader
def load_dataset(batch_size:int, mode:'str') -> tuple :
    img_tf_eval = torchvision.transforms.Compose(
        transforms=[
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ]
    )
    if mode == "train" :
        img_tf_train = torchvision.transforms.Compose(
            transforms=[
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
            ]
        )
        train = torchvision.datasets.CIFAR10(root='data', download=True, train=True, transform=img_tf_train)        
    elif mode == 'eval' :
        train = torchvision.datasets.CIFAR10(root='data', download=True, train=True, transform=img_tf_eval)
    test = torchvision.datasets.CIFAR10(root='data', download=True, train=False, transform=img_tf_eval)
    train_loader = DataLoader(dataset=train, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False, num_workers=0)
    output = (train_loader, test_loader)
    return output