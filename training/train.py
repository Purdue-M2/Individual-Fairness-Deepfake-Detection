import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import numpy as np
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
from log_utils import Logger
import os.path as osp
import sys
import pandas as pd
from utils.bypass_bn import enable_running_stats, disable_running_stats
from sam import SAM
from PIL import Image
from collections import OrderedDict
from networks.anchor_xcep import xception as anchor_xcep
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.distributions.bernoulli import Bernoulli
from networks.xception import Xception
from networks.resnet50 import ResNet50
from loss.ind_loss import compute_l1_loss, compute_l2_loss
#from transform import xception_default_data_transforms as data_transforms #for efficient_naive, resnet_naive
from transform import fair_df_default_data_transforms as data_transforms #for xception_naive, xception_ours, efficient_ours, resnet_ours
#from transform import get_albumentations_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageDataset(Dataset):
    def __init__(self, csv_file, owntransforms):
        super(ImageDataset, self).__init__()
        self.img_path_label = pd.read_csv(csv_file)
        self.transform = owntransforms

    def __len__(self):
        return len(self.img_path_label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_path_label.iloc[idx, 0]

        if img_path != 'img_path':
            img = Image.open(img_path)
            img = self.transform(img)

            label = np.array(self.img_path_label.loc[idx, 'label'])
            intersec_label = np.array(self.img_path_label.loc[idx, 'intersec_label'])
        return {'image': img, 'label': label, 'intersec_label': intersec_label}
    
def adjust_state_dict(state_dict):
    """
    Remove 'module.' prefix from the state_dict if the model was trained using DistributedDataParallel.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "") if key.startswith("module.") else key
        new_state_dict[new_key] = value
    return new_state_dict

def xcept_pretrained_model(num, adjust_channel=True):
    model = anchor_xcep(num_classes=1000, pretrained='imagenet', adjust_channel=adjust_channel)
    numFtrs = model.last_linear.in_features
    model.last_linear = nn.Linear(numFtrs, num)
    return model

def res_pretrained_model(numClasses):
    model = ResNet50()
    if(opt.mode == 'ours'):
        #modify input layer to accept 6 channels
        original_conv1 = model.conv1
        new_conv1 = nn.Conv2d(
            in_channels=6,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
        with torch.no_grad():
            # Copy pretrained weights for channels 0–2
            new_conv1.weight[:, :3] = original_conv1.weight
            # Duplicate them for channels 3–5
            new_conv1.weight[:, 3:] = original_conv1.weight
        model.conv1 = new_conv1
    model.fc = nn.Linear(512, numClasses)
    return model

def eff_pretrained_model(numClasses):
    model = EfficientNet.from_pretrained('efficientnet-b3')

    if(opt.mode == 'ours'):
        # Modify first conv layer to accept 6 channels
        original_weights = model._conv_stem.weight.clone()
        model._conv_stem = nn.Conv2d(6, 40, kernel_size=3, stride=2, padding=1, bias=False)

        # Initialize first 3 channels with pre-trained weights, copy them to the extra 3 channels
        model._conv_stem.weight.data[:, :3, :, :] = original_weights
        model._conv_stem.weight.data[:, 3:, :, :] = original_weights  

    # Replace the final classifier layer
    model._fc = nn.Linear(1536, numClasses)

    return model

def sample_reference(x, reference_set):
    """
    Samples a reference image from the reference set ensuring it is different from x.
    
    Args:
        x (torch.Tensor): Current batch of input images (B, C, H, W).
        reference_set (torch.Tensor): Precomputed reference images (N, C, H, W).
        
    Returns:
        torch.Tensor: Sampled reference images (B, C, H, W).
    """
    batch_size = x.shape[0]
    reference_set = reference_set.to(x.device)
    ref_indices = torch.randint(0, len(reference_set), (batch_size,))
    
    # Ensure sampled references are not exactly the same as x
    for i in range(batch_size):
        while torch.equal(reference_set[ref_indices[i]], x[i]):
            ref_indices[i] = torch.randint(0, len(reference_set), (1,))
    
    return reference_set[ref_indices].to(x.device)  # Ensure same device

def sample_fixed_reference(x, reference_set):
    """
    Selects one random reference from the reference set for each test sample.

    Args:
        x (torch.Tensor): Batch of test images (B, C, H, W).
        reference_set (torch.Tensor): Set of stored reference images (N, C, H, W).

    Returns:
        torch.Tensor: Selected reference images (B, C, H, W).
    """
    batch_size = x.shape[0]
    # Select one random reference per test sample
    ref_indices = torch.randint(0, len(reference_set), (batch_size,))
    selected_refs = reference_set[ref_indices].to(x.device)  # Ensure the reference is on the correct device
    return selected_refs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_batchsize", type=int,
                        default=32, help="size of the test batches")
    parser.add_argument("--train_batchsize", type=int,
                        default=32, help="size of the train batches")
    parser.add_argument("--checkpoints", type=str, default = '', help="continue train model path")
    parser.add_argument("--mode", type=str, default = 'naive', help="naive model or ours")
    parser.add_argument("--model", type=str, default = 'xception', help="xception, efficientnet, resnet")


    opt = parser.parse_args()
    print(opt, '!!!!!!!!!!!')
    sys.stdout = Logger(osp.join('./checkpoints_anchor/'+'mask'+'/log_training.txt'))

    cuda = True if torch.cuda.is_available() else False
    
    if(opt.model == 'xception'):
        if(opt.mode == 'ours'):
            model = xcept_pretrained_model(1, adjust_channel=True) #adjust num_classes as needed 
        else:
            xception_config = {
                "num_classes": 1,  # Binary classification
                "mode": "normal",  # or "shallow_xception" if needed
                "inc": 3,         
                "dropout": 0.5    # Dropout rate
            }
            model = Xception(xception_config) 
    elif(opt.model == 'efficientnet'):
        model = eff_pretrained_model(1) #adjust num_classes as needed 
    elif(opt.model == 'resnet'):
        model = res_pretrained_model(1) #adjust num_classes as needed 
    
    model.to(device)
    if opt.checkpoints != '':
        print('Loading checkpoint from:', opt.checkpoints)
        checkpoint = torch.load(opt.checkpoints, map_location='cuda:0')
        model.load_state_dict(checkpoint)
    # criterion = nn.BCEWithLogitsLoss(reduction='none').to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)

    params_to_update = model.parameters()
    base_optimizer = torch.optim.SGD
    optimizer = SAM(params_to_update, base_optimizer,
                    lr=0.0005, momentum=0.9, weight_decay=5e-03)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-03)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.9)

    
    train_dataset = ImageDataset('./ff++/train.csv', data_transforms['train'])
    test_dataset = ImageDataset('./ff++/test.csv', data_transforms['test'])
    
    # Define a reference set that is a subset of the training dataset
    reference_indices = torch.randperm(len(train_dataset))[:5000]  # Randomly select 5000 samples
    reference_set = [train_dataset[i]['image'] for i in reference_indices]  # Extract images
    reference_set = torch.stack(reference_set)  # Convert list to tensor
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=opt.train_batchsize, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=opt.test_batchsize, shuffle=False, num_workers=8, pin_memory=True)
    
    alpha = 0.2
    tau = 0.00001 
    lambda_val = 0.001 
    running_corrects = 0
    total_samples = 0
    
    for epoch in range(50):
        print('Epoch {}/{}'.format(epoch, 100 - 1))
        model.train()
        total_loss = 0.0
        total_ind_loss_sum = 0.0
        for i, data_dict in enumerate(tqdm(train_dataloader)):
            data, label = data_dict['image'], data_dict["label"]
            data, label = data.to(device), label.to(device)
            
            mask = Bernoulli(torch.tensor([alpha])).sample().bool().item()

            # Sample reference r from reference set, ensuring it is not the same as x
            r = sample_reference(data, reference_set)

            if mask:
                anc = torch.cat([torch.zeros_like(data), data - r], dim=1)  # Reference masked
                target = torch.full_like(label, 0.5)  # Uniform probability
            else:
                anc = torch.cat([r, data - r], dim=1)  # Standard anchoring
                target = label.float()
            
            enable_running_stats(model)
            if(opt.model == 'xception'):
                if(opt.mode == 'ours'):
                    outputs, _ = model(anc) #for xception ours model
                else:
                    outputs, _ = model(data) #for xception naive model
            else:
                if(opt.mode == 'ours'):
                    outputs = model(anc) #for resnet & efficientnet ours model
                else:
                    outputs = model(data) #for resnet & efficientnet naive models
            outputs.to(device)
            preds = (outputs.squeeze(1).sigmoid()) >= 0.5
            if(opt.mode=='ours'):
                ce_loss, independence_loss = compute_l2_loss(outputs, data, label, tau, criterion)
            else:
                ce_loss, independence_loss = compute_l1_loss(outputs, data, label, tau, criterion)
            loss = ce_loss + lambda_val*independence_loss
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            disable_running_stats(model)
            if(opt.model == 'xception'):
                if(opt.mode == 'ours'):
                    outputs, _ = model(anc) #for xception ours model
                else:
                    outputs, _ = model(data) #for xception naive model
            else:
                if(opt.mode == 'ours'):
                    outputs = model(anc) #for resnet & efficientnet ours model
                else:
                    outputs = model(data) #for resnet & efficientnet naive models
            outputs.to(device)
            if(opt.mode=='ours'):
                ce_loss, independence_loss = compute_l2_loss(outputs, data, label, tau, criterion)
            else:
                ce_loss, independence_loss = compute_l1_loss(outputs, data, label, tau, criterion)
            total_ind_loss_sum += independence_loss.item()
            loss = ce_loss + lambda_val*independence_loss
            loss.backward()
            optimizer.second_step(zero_grad=True)

            batch_correct = (preds == label).sum().item()
            batch_total = label.size(0)
            batch_acc = batch_correct / batch_total
            
            running_corrects += batch_correct
            total_samples += batch_total
            running_acc = running_corrects / total_samples
            if i % 50 == 0:
                print(f"\nBatch {i} metrics:")
                print(f"Batch Accuracy: {batch_acc:.4f} ({batch_correct}/{batch_total})")
                print(f"Running Accuracy: {running_acc:.4f} ({running_corrects}/{total_samples})")
                average_ind_loss = total_ind_loss_sum / len(test_dataloader)
                print(f"Total Ind Loss: {total_ind_loss_sum:.4f}")
                print(f"Average Ind Loss: {average_ind_loss:.4f}")
        scheduler.step()
        # evaluation

        if (epoch+1) % 1 == 0:

            savepath = f'./checkpoints_anchor/xception/{lambda_val}_{tau}'
            temp_model = savepath+"/"+str(epoch)+'.pth'
            torch.save(model.state_dict(), temp_model)
            print(temp_model)

            print()
            print('-' * 10)
            pred_list = []
            label_list = []
            total_ind_loss_sum = 0

            # Load a fixed reference set (precomputed from training data)
            reference_set = torch.stack([test_dataset[i]['image'] for i in range(500)])  # Example: 500 reference images


            for i, data_dict in enumerate(tqdm(test_dataloader)):

                model.eval()
                data, label = data_dict['image'], data_dict["label"]
                if 'label_spe' in data_dict:
                    data_dict.pop('label_spe')  
                data_dict['image'], data_dict["label"] = data.to(device), label.to(device)

                with torch.no_grad():
                    # output = model(data_dict, inference=True)
                    
                    batch_size = data_dict['image'].shape[0]

                    # Select a single random reference from the batch itself
                    random_idx = torch.randint(0, batch_size, (1,))  # Pick one random index
                    r = data_dict['image'][random_idx].repeat(batch_size, 1, 1, 1)  # Repeat for the whole batch

                    # Compute residuals
                    residual = data_dict['image'] - r
                    anchored_input = torch.cat([r, residual], dim=1)  # Shape: [batch, 6, H, W]
                    
                    # Forward pass
                    if(opt.model == 'xception'):
                        if(opt.mode == 'ours'):
                            output, _ = model(anchored_input) #for xception ours model
                        else:
                            output, _ = model(data_dict['image']) #for xception naive model
                    else:
                        if(opt.mode == 'ours'):
                            output = model(anchored_input) #for resnet & efficientnet ours model
                        else:
                            output = model(data_dict['image']) #for resnet & efficientnet naive models
                    ce_loss, independence_loss = compute_l2_loss(output, data_dict['image'], data_dict["label"], tau, criterion)
                
                    total_ind_loss_sum += independence_loss.item()

                    pred = output  
                    pred_list += pred.cpu().numpy().tolist()
                    label_list += label.cpu().numpy().tolist()

    
            label_list = np.array(label_list)
            pred_list = torch.tensor(pred_list)

            pred_probs = torch.sigmoid(pred_list).numpy()

            average_ind_loss = total_ind_loss_sum / len(test_dataloader)
            print(f"Average Ind Loss: {average_ind_loss:.5f}")

            auc = roc_auc_score(label_list, pred_probs)
            print(f"AUC: {auc:.5f}")




