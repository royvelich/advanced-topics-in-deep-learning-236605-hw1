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


def list_subdirectories(base_dir='.'):
    result = []
    for current_sub_dir in os.listdir(base_dir):
        full_sub_dir_path = os.path.join(base_dir, current_sub_dir)
        if os.path.isdir(full_sub_dir_path):
            result.append(full_sub_dir_path)

    return result


def get_latest_subdirectory(base_dir='.'):
    subdirectories = list_subdirectories(base_dir)
    return os.path.normpath(max(subdirectories, key=os.path.getmtime))


if __name__ == '__main__':
    #####################################
    # Download Dataset
    #####################################

    # https://github.com/fastai/imagenette
    dataset_url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz'
    dataset_filename = dataset_url.split('/')[-1]
    dataset_foldername = dataset_filename.split('.')[0]
    data_path = './data'
    dataset_filepath = os.path.join(data_path, dataset_filename)
    dataset_folderpath = os.path.join(data_path, dataset_foldername)

    os.makedirs(data_path, exist_ok=True)

    download = False
    if not os.path.exists(dataset_filepath):
        download = True
    else:
        md5_hash = hashlib.md5()

        file = open(dataset_filepath, "rb")

        content = file.read()

        md5_hash.update(content)

        digest = md5_hash.hexdigest()
        if digest != 'fe2fc210e6bb7c5664d602c3cd71e612':
            download = True
    if download:
        download_url(dataset_url, data_path)

    with tarfile.open(dataset_filepath, 'r:gz') as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=data_path)

    #####################################
    # Create DataLoader
    #####################################
    size = 224
    ks = (int(0.1 * size) // 2) * 2 + 1  # should be odd
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    train_transform_classifier = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**__imagenet_stats)])

    val_transform_classifier = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**__imagenet_stats)])

    dataset_train_classifier = torchvision.datasets.ImageFolder(os.path.join(dataset_folderpath, 'train'), train_transform_classifier)
    dataset_val_classifier = torchvision.datasets.ImageFolder(os.path.join(dataset_folderpath, 'val'), val_transform_classifier)

    # settings
    classifier_train_epochs = 700
    batch_size = 220
    t = 0.07
    momentum = 0.9
    weight_decay = 1e-4
    lr = 1e-3
    k = 10000
    checkpoint_granularity = 50

    train_dataloader_classifier = torch.utils.data.DataLoader(
            dataset_train_classifier,
            batch_size=batch_size,
            num_workers=20,
            drop_last=True,
            shuffle=True,
            pin_memory=True,
            prefetch_factor=20)

    val_dataloader_classifier = torch.utils.data.DataLoader(
            dataset_val_classifier,
            batch_size=batch_size,
            num_workers=20,
            drop_last=True,
            shuffle=True,
            pin_memory=True,
            prefetch_factor=20)

    #####################################
    # Train
    #####################################

    moco = MoCo(k=k).cuda()
    classifier = LinearClassifier(c=10).cuda()

    latest_subdir = get_latest_subdirectory(results_base_path)
    moco.load_state_dict(torch.load(os.path.join(latest_subdir, 'moco_950.pt'), map_location=torch.device('cuda')))

    results_dir_path = latest_subdir
    Path(results_dir_path).mkdir(parents=True, exist_ok=True)
    results_filepath = os.path.normpath(os.path.join(results_dir_path, 'classifier_loss_results.npy'))

    sys.stdout = Logger(filepath=os.path.join(results_dir_path, 'linear_classifier_training.log'))

    # ----------------
    # Train classifier
    # ----------------
    classifier_train_loss_array = np.array([])
    classifier_acc1_array = np.array([])
    classifier.freeze_features(moco.get_query_encoder())
    classifier_loss = torch.nn.CrossEntropyLoss().cuda()
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    for classifier_epoch_index in range(classifier_train_epochs):

        # --------------------------------------
        # Train classifier for one epoch
        # --------------------------------------
        classifier_batch_loss_array = np.array([])
        classifier_epoch_loss = 0
        classifier.train()
        print(f'Classifier Training Epoch #{classifier_epoch_index + 1}')
        print('------------------------------------------')
        for classifier_batch_index, (images, labels) in enumerate(train_dataloader_classifier):
            logits = classifier.forward(images.cuda())
            loss_output = classifier_loss(logits, labels.cuda())
            classifier_optimizer.zero_grad()
            loss_output.backward()
            classifier_optimizer.step()
            loss_item = loss_output.item()
            classifier_batch_loss_array = np.append(classifier_batch_loss_array, [loss_item])
            classifier_epoch_loss = np.mean(classifier_batch_loss_array)
            print(f'Epoch: #{(classifier_epoch_index + 1):{" "}{"<"}{5}}| Batch: #{(classifier_batch_index + 1):{" "}{"<"}{5}}| Batch Loss: {loss_output.item():{" "}{"<"}{30}}| Epoch Loss: {classifier_epoch_loss:{" "}{"<"}{30}}')
        print('')
        classifier_train_loss_array = np.append(classifier_train_loss_array, [classifier_epoch_loss])

        print(f'Classifier Validation Epoch #{classifier_epoch_index + 1}')
        print('------------------------------------------')
        classifier.eval()
        acc1_array = np.array([])
        acc1 = 0
        for classifier_batch_index, (images, labels) in enumerate(val_dataloader_classifier):
            logits = classifier.forward(images.cuda())
            _, pred = logits.topk(k=1, dim=1, largest=True, sorted=True)
            correct_preds = float(pred.squeeze().eq(labels.cuda()).sum())
            acc1_array = np.append(acc1_array, [correct_preds / float(labels.shape[0])])
            acc1 = acc1_array.mean()
            print(f'Batch: #{(classifier_batch_index + 1):{" "}{"<"}{5}}| Top-1 Accuracy: {acc1:{" "}{"<"}{30}}')
        print('')
        classifier_acc1_array = np.append(classifier_acc1_array, [acc1])

        # Save current loss results
        results = {
            'classifier_train_loss_array': classifier_train_loss_array,
            'classifier_train_epochs': classifier_train_epochs,
            'classifier_acc_array': classifier_acc1_array,
        }
        np.save(file=results_filepath, arr=results, allow_pickle=True)

        # Save classifier model
        torch.save(classifier.state_dict(), os.path.normpath(os.path.join(results_dir_path, f'classifier_{classifier_epoch_index + 1}.pt')))
