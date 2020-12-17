import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
from torch.autograd import Variable

num_epochs = 150
batch_size = 16
GPU_NUM = 0

#class_weight = torch.Tensor([0.317, 0.683]).cuda()

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C

        target = target.view(-1, 1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def train_model(model, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                #loss = criterion(outputs, labels)
                loss = criterion.forward(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = 100 * (running_corrects.double() / len(image_datasets[phase]))
            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            torch.save(model, args.output + '/Resnet50_{0:003}.pth' .format(epoch))

    return model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True,
                  help='path to folder train')
    ap.add_argument('--valid', required=True,
                  help='path to folder valid')
    ap.add_argument('-o','--output', required=True,
                  help='path to save model trained')
    args = ap.parse_args()
    train_path = args.train
    valid_path = args.valid if args.valid else train_path


    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                  std=[0.2023, 0.1994, 0.2010])

    data_transforms = {
    'train':
        transforms.Compose([
            transforms.Resize((224,224), interpolation=Image.NEAREST),
            # transforms.RandomGrayscale(p=0.1),
            # transforms.RandomAffine(0, shear=5, scale=(0.8,1.2)),
            # transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.8, 1.5), saturation=0),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
    'validation':
        transforms.Compose([
        transforms.Resize((224,224), interpolation=Image.NEAREST),
        transforms.ToTensor(),
        normalize
        ]),
    }

    image_datasets = {
      'train':
      datasets.ImageFolder(train_path, data_transforms['train']),
      'validation':
      datasets.ImageFolder(valid_path, data_transforms['validation'])
    }

    dataloaders = {
      'train':
      torch.utils.data.DataLoader(image_datasets['train'],
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=4),
      'validation':
      torch.utils.data.DataLoader(image_datasets['validation'],
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4)
    }
    class_names = image_datasets['train'].classes

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
    print('-' * 40)
    print(class_names)
    print(dataset_sizes)
    print('-' * 40)

    inputs, classes = next(iter(dataloaders['train']))
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device ', torch.cuda.current_device())

    if device.type == 'cuda':
        print(torch.cuda.get_device_name(GPU_NUM))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(GPU_NUM) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(GPU_NUM) / 1024 ** 3, 1), 'GB')

    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    #model_ft.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, len(class_names)))
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)
    model_ft.eval()

    #criterion = nn.CrossEntropyLoss(weight=class_weight).to(device)
    criterion = FocalLoss(gamma=2).to(device)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)

    # Decay LR by a factor of 0.1 every step_size epochsScratch
    # Final Thoughts and Where to Go Next
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    model_trained = train_model(model_ft, criterion, optimizer_ft, num_epochs=num_epochs)