from torchvision.datasets.utils import download_url
import os
import tarfile
import hashlib
import torchvision
import torch
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
from augmentation import TwoCropsTransform
from moco import KeyEncoder
from moco import QueryEncoder
from moco import MoCo

# https://github.com/fastai/imagenette
dataset_url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz'
dataset_filename = dataset_url.split('/')[-1]
dataset_foldername = dataset_filename.split('.')[0]
data_path = './data'
dataset_filepath = os.path.join(data_path,dataset_filename)
dataset_folderpath = os.path.join(data_path,dataset_foldername)

if __name__ == '__main__':

    #####################################
    # Download Dataset
    #####################################

    # # https://github.com/fastai/imagenette
    # dataset_url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz'
    # dataset_filename = dataset_url.split('/')[-1]
    # dataset_foldername = dataset_filename.split('.')[0]
    # data_path = './data'
    # dataset_filepath = os.path.join(data_path, dataset_filename)
    # dataset_folderpath = os.path.join(data_path, dataset_foldername)
    #
    # os.makedirs(data_path, exist_ok=True)
    #
    # download = False
    # if not os.path.exists(dataset_filepath):
    #     download = True
    # else:
    #     md5_hash = hashlib.md5()
    #
    #     file = open(dataset_filepath, "rb")
    #
    #     content = file.read()
    #
    #     md5_hash.update(content)
    #
    #     digest = md5_hash.hexdigest()
    #     if digest != 'fe2fc210e6bb7c5664d602c3cd71e612':
    #         download = True
    # if download:
    #     download_url(dataset_url, data_path)
    #
    # with tarfile.open(dataset_filepath, 'r:gz') as tar:
    #     tar.extractall(path=data_path)

    #####################################
    # Create DataLoader
    #####################################
    size = 224
    ks = (int(0.1 * size) // 2) * 2 + 1  # should be odd
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    train_transform = TwoCropsTransform(transforms.Compose([transforms.RandomResizedCrop(scale=(0.2, 1), size=size),
                                                       transforms.RandomHorizontalFlip(),
                                                       transforms.RandomApply(
                                                                [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                                                       transforms.RandomGrayscale(p=0.2),
                                                       # transforms.GaussianBlur(kernel_size=ks),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(**__imagenet_stats)]))

    dataset_train = torchvision.datasets.ImageFolder(os.path.join(dataset_folderpath, 'train'), train_transform)

    # settings
    epochs = 50
    batch_size = 128
    t = 0.07
    momentum = 0.9
    weight_decay = 1e-4
    lr = 0.005
    k = 10000

    train_dataloader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            num_workers=20,
            drop_last=True,
            shuffle=True,
            pin_memory=True,
            prefetch_factor=20
    )

    # def get_numpy_samples(inputs):
    #     mean = torch.as_tensor(__imagenet_stats['mean'], dtype=inputs.dtype, device=inputs.device)
    #     std = torch.as_tensor(__imagenet_stats['std'], dtype=inputs.dtype, device=inputs.device)
    #     inputs = inputs * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
    #     inputs = inputs.numpy()
    #     inputs = np.transpose(inputs, (0, 2, 3, 1))
    #     return inputs

    # fig, axes = plt.subplots(nrows=batch_size, ncols=2, figsize=(10, 100))
    # for (input1, input2), _ in train_dataloader:
    #     np_inputs1, np_inputs2 = get_numpy_samples(input1), get_numpy_samples(input2)
    #     for row in range(batch_size):
    #         axes[row, 0].axis("off")
    #         axes[row, 0].imshow(np_inputs1[row])
    #         axes[row, 1].axis("off")
    #         axes[row, 1].imshow(np_inputs2[row])
    #     break
    # plt.show()

    #####################################
    # Train Feature Extractor
    #####################################

    moco = MoCo(k=k)
    loss = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        moco.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay)

    moco.train()

    for epoch in range(epochs):
        loss_array = np.array([])
        for batch_index, ((queries, keys), _) in enumerate(train_dataloader):
            logits, labels, queries_features, keys_features = moco.forward(queries=queries.cuda(), keys=keys.cuda())
            output = loss(logits/t, labels)
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
            moco.update_key_encoder(keys=keys_features)
            loss_item = output.item()
            loss_array = np.append(loss_array, [loss_item])
            print(f'Epoch: #{epoch+1}; Batch: #{batch_index+1}; Batch Loss: {output.item()}; Average Loss: {np.mean(loss_array)}')
