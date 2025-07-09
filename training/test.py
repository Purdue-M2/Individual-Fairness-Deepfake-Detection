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
        original_conv1 = model.resnet[0]  # conv1 is the first layer in torchvision's resnet
        model.resnet[0] = nn.Conv2d(
            in_channels=6,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
        with torch.no_grad():
            model.resnet[0].weight[:, :3] = original_conv1.weight  # Copy RGB weights
            model.resnet[0].weight[:, 3:] = original_conv1.weight[:, :3]
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
                        default=32, help="size of the batches")
    parser.add_argument("--train_batchsize", type=int,
                        default=32, help="size of the batches")
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
        new_state_dict = OrderedDict()
        
        for k, v in checkpoint.items():
            name = k.replace("module.", "")  # Remove "module." prefix
            new_state_dict[name] = v
        
        model.load_state_dict(checkpoint['model_state_dict']) #for xception & resnet naive checkpoints
        #model.load_state_dict(new_state_dict) #for efficient naive checkpoint
        #model.load_state_dict(checkpoint) #for xception, resnet, efficient ours checkpoint

    # criterion = nn.BCEWithLogitsLoss(reduction='none').to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    params_to_update = model.parameters()
    optimizer = optim.SGD(params_to_update, 0.0005, momentum=0.9, weight_decay=5e-03)
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
    tau = 0.00005
    lambda_val = 0.001
    running_corrects = 0
    total_samples = 0
    
        
    pred_list = []
    label_list = []
    total_ind1_loss_sum = 0
    total_ind2_loss_sum = 0

    # Load a fixed reference set (precomputed from training data)
    reference_set = torch.stack([test_dataset[i]['image'] for i in range(500)])  # Example: 500 reference images


    for i, data_dict in enumerate(tqdm(test_dataloader)):

        model.eval()
        data, label = data_dict['image'], data_dict["label"]
        if 'label_spe' in data_dict:
            data_dict.pop('label_spe')  # remove the specific label
        data_dict['image'], data_dict["label"] = data.to(device), label.to(device)

        with torch.no_grad():
            # output = model(data_dict, inference=True)

            
            batch_size = data_dict['image'].shape[0]
            random_idx = torch.randint(0, batch_size, (1,))  # Pick one random index
            r = data_dict['image'][random_idx].repeat(batch_size, 1, 1, 1)  # Repeat for the whole batch
            residual = data_dict['image'] - r
            anchored_input = torch.cat([r, residual], dim=1)  # Shape: [batch, 6, H, W]
            
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
            ce_loss, independence1_loss = compute_l1_loss(output, data_dict['image'], data_dict["label"], tau, criterion)
        
            total_ind1_loss_sum += independence1_loss.item()
            ce_loss, independence2_loss = compute_l2_loss(output, data_dict['image'], data_dict["label"], tau, criterion)
        
            total_ind2_loss_sum += independence2_loss.item()

            pred = output  # assuming pred is the logits
            pred_list += pred.cpu().numpy().tolist()
            label_list += label.cpu().numpy().tolist()


    # Convert lists to numpy arrays
    label_list = np.array(label_list)
    pred_list = torch.tensor(pred_list)

    # Use torch.sigmoid to convert logits to probabilities
    pred_probs = torch.sigmoid(pred_list).numpy()

    average_ind1_loss = total_ind1_loss_sum / len(test_dataloader)
    average_ind2_loss = total_ind2_loss_sum / len(test_dataloader)
    print(f"Average Ind1 Loss: {average_ind1_loss:.5f}")
    print(f"Average Ind2 Loss: {average_ind2_loss:.5f}")
    # Calculate AUC score
    auc = roc_auc_score(label_list, pred_probs)
    print(f"AUC: {auc:.5f}")




