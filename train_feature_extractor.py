from torchvision.datasets.utils import download_url
import os
import sys
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
from moco import LinearClassifier
from datetime import datetime
from pathlib import Path
from moco import MoCo
from logger import Logger

# https://github.com/fastai/imagenette
dataset_url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz'
dataset_filename = dataset_url.split('/')[-1]
dataset_foldername = dataset_filename.split('.')[0]
data_path = './data'
results_base_path = './results'
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

    train_transform_features_extractor = TwoCropsTransform(transforms.Compose([
        transforms.RandomResizedCrop(scale=(0.2, 1), size=size),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=ks)]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**__imagenet_stats)]))

    dataset_train_features_extractor = torchvision.datasets.ImageFolder(os.path.join(dataset_folderpath, 'train'), train_transform_features_extractor)

    # settings
    t = 0.07
    feature_extractor_train_epochs = 2000
    feature_extractor_train_batch_size = 220
    classifier_train_epochs = 5
    momentum = 0.9
    weight_decay = 1e-4
    lr = 1e-4
    k = 10000
    checkpoint_granularity = 50

    train_dataloader_features_extractor = torch.utils.data.DataLoader(
            dataset_train_features_extractor,
            batch_size=feature_extractor_train_batch_size,
            num_workers=20,
            drop_last=True,
            shuffle=True,
            pin_memory=True,
            prefetch_factor=20)

    #####################################
    # Train
    #####################################

    moco = MoCo(k=k).cuda()
    moco.train()
    loss_features_extractor = torch.nn.CrossEntropyLoss().cuda()
    optimizer_features_extractor = torch.optim.Adam(
        moco.parameters(),
        lr=lr)

    results_dir_path = os.path.normpath(os.path.join(results_base_path, datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
    Path(results_dir_path).mkdir(parents=True, exist_ok=True)
    results_filepath = os.path.normpath(os.path.join(results_dir_path, 'feature_extractor_loss_results.npy'))
    sys.stdout = Logger(filepath=os.path.join(results_dir_path, 'feature_extractor_training.log'))

    feature_extractor_train_loss_array = np.array([])
    for epoch_index in range(feature_extractor_train_epochs):

        # --------------------------------------
        # Train features extractor for one epoch
        # --------------------------------------
        feature_extractor_batch_loss_array = np.array([])
        feature_extractor_epoch_loss = 0
        print(f'Feature Extractor Epoch #{epoch_index + 1}')
        print('------------------------------------------')
        for batch_index, ((queries, keys), _) in enumerate(train_dataloader_features_extractor):
            logits, labels, queries_features, keys_features = moco.forward(queries=queries.cuda(), keys=keys.cuda())
            loss_output = loss_features_extractor(logits / t, labels)
            optimizer_features_extractor.zero_grad()
            loss_output.backward()
            optimizer_features_extractor.step()
            moco.update_key_encoder(keys=keys_features)
            loss_item = loss_output.item()
            feature_extractor_batch_loss_array = np.append(feature_extractor_batch_loss_array, [loss_item])
            feature_extractor_epoch_loss = np.mean(feature_extractor_batch_loss_array)
            print(f'Epoch: #{(epoch_index + 1):{" "}{"<"}{5}}| Batch: #{(batch_index + 1):{" "}{"<"}{5}}| Batch Loss: {loss_output.item():{" "}{"<"}{30}}| Epoch Loss: {feature_extractor_epoch_loss:{" "}{"<"}{30}}')
        print('')
        feature_extractor_train_loss_array = np.append(feature_extractor_train_loss_array, [feature_extractor_epoch_loss])

        if (epoch_index + 1) % checkpoint_granularity == 0:
            results = {
                'feature_extractor_train_loss_array': feature_extractor_train_loss_array,
                'feature_extractor_train_epochs': feature_extractor_train_epochs,
                'feature_extractor_train_batch_size': feature_extractor_train_batch_size
            }

            np.save(file=results_filepath, arr=results, allow_pickle=True)
            lastest_moco_filepath = os.path.normpath(os.path.join(results_dir_path, f'moco_{epoch_index + 1}.pt'))
            torch.save(moco.state_dict(), lastest_moco_filepath)
