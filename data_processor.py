import torchvision
import torch


def data_loader(args):
    kwopt = {'num_workers': 8, 'pin_memory': True}
    trn_transforms = torchvision.transforms.Compose([
        #                     torchvision.transforms.CenterCrop(img_size),
        torchvision.transforms.RandomCrop(args.image_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        #                     torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((480, 320)),
        torchvision.transforms.ToTensor(),
        #                     torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    trn_dataset = torchvision.datasets.ImageFolder('./BSDS500/train', transform=trn_transforms)
    val_dataset = torchvision.datasets.ImageFolder('./BSDS500/val', transform=val_transforms)
    trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True, **kwopt, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, **kwopt, drop_last=False)
    return trn_loader, val_loader
