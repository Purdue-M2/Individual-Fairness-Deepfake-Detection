
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2
xception_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}

fair_df_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}


resnet_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}

def get_albumentations_transforms(use_methods):
    transforms_list = [A.Resize(256, 256)]
    if 'gaussian_blur' in use_methods:
        transforms_list.append(A.GaussianBlur(blur_limit=(3, 3), p=1.0))
        print('gaussian_blur kernel size 5')
    if 'jpeg_compression' in use_methods:
        transforms_list.append(A.ImageCompression(compression_type='jpeg', quality_range=(90, 90), p=1.0))
        print('using jpeg compression 100')
    if 'random_crop' in use_methods:
        transforms_list.append(A.RandomCrop(height=224, width=224, p=1.0))
        print('random_crop_224')
    if 'center_crop' in use_methods:
        transforms_list.append(A.CenterCrop(height=224, width=224, p=1.0))
    if 'hue_saturation_value' in use_methods:
        transforms_list.append(A.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=50, val_shift_limit=50, p=1.0))
        print('HueSaturationValue applied with a hue shift limit of 50, saturation shift limit of 50, and value shift limit of 50.')
    if 'random_brightness_contrast' in use_methods:
        transforms_list.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0))
        print('BrightnessContrast applied with a brightness limit of 0.8 and a contrast limit of 0.8.')
    if 'rotation' in use_methods:
        transforms_list.append(A.Rotate(limit=60, p=1.0))
        print('rotation applied with limit 15')

    # Converting the image to PyTorch tensor and normalize
    transforms_list.extend([
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])

    return A.Compose(transforms_list)
