import torch
import torch.nn.functional as F

def divide_and_shuffle_patches(imgs, patch_size=32):
    """
    random patch shuffle per image
    """
    
    B, C, H, W = imgs.shape
    
    if patch_size > H or patch_size > W:
        raise ValueError(f"Patch size {patch_size} is larger than input dimensions ({H}, {W}).")
    
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    if pad_h > 0 or pad_w > 0:
        imgs = F.pad(imgs, (0, pad_w, 0, pad_h), mode='reflect')
    
    H_pad, W_pad = imgs.shape[2], imgs.shape[3]
    n_h, n_w = H_pad // patch_size, W_pad // patch_size
    total_patches = n_h * n_w

    patches = imgs.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    
    patches = patches.contiguous().view(B, total_patches, C, patch_size, patch_size)

    for b in range(B):
        idx = torch.randperm(total_patches, device=imgs.device)
        patches[b] = patches[b][idx]

    patches = patches.view(B, n_h, n_w, C, patch_size, patch_size)
    reconstructed = patches.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H_pad, W_pad)

    if pad_h > 0 or pad_w > 0:
        reconstructed = reconstructed[:, :, :H, :W]
    
    return reconstructed

def gaussian(imgs, kernel_size = 3, sigma = 1.0):
    padding = (kernel_size - 1) // 2

    # Create Gaussian kernel
    x = torch.arange(kernel_size, device=imgs.device).float()
    xy_grid = torch.stack(torch.meshgrid(x, x), dim=-1)
    mean = (kernel_size - 1) / 2
    variance = sigma ** 2
    gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2, dim=-1) / (2 * variance))
    gaussian_kernel /= gaussian_kernel.sum()
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).repeat(imgs.size(1), 1, 1, 1)

    # Apply Gaussian filter
    gaussian_filter = torch.nn.Conv2d(
        in_channels=imgs.size(1),
        out_channels=imgs.size(1),
        kernel_size=kernel_size,
        groups=imgs.size(1),
        bias=False,
        padding=padding
    ).to(imgs.device)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter(imgs)

def process_image_batch(imgs):
    shuffled_imgs = divide_and_shuffle_patches(imgs)
    filtered_imgs = gaussian(shuffled_imgs)
    residuals = shuffled_imgs - filtered_imgs
    fft_results = torch.fft.rfft2(residuals)
    fft_magnitudes = torch.abs(fft_results)
    fft_magnitudes = 20 * torch.log1p(fft_magnitudes + 1e-8)
    fft_magnitudes = 255 * (fft_magnitudes / fft_magnitudes.max())
    return fft_magnitudes

def compute_l1_loss(logits, imgs, labels, tau, criterion):
    ce_loss = criterion(logits.squeeze(1), labels.float())
    batch_size = imgs.size(0)
    logits = logits.squeeze(1)
    outputs_diff = logits.unsqueeze(1) - logits.unsqueeze(0)  
    outputs_dist = torch.abs(outputs_diff) 
    # Flatten feature dims and compute L2 norm
    imgs = imgs.view(imgs.size(0), -1)
    freq_diff = torch.cdist(imgs, imgs, p=2)
    # Compute hinge loss matrix
    hinge_matrix = torch.relu(outputs_dist - tau * freq_diff)
    independence_loss = torch.sum(torch.triu(hinge_matrix, diagonal=1))
    # Normalize by number of pairs
    num_pairs = (batch_size * (batch_size - 1)) // 2
    independence_loss = independence_loss / num_pairs if num_pairs > 0 else 0.0
    return ce_loss, independence_loss

def compute_l2_loss(logits, imgs, labels, tau, criterion):
    ce_loss = criterion(logits.squeeze(1), labels.float())
    freq_features = process_image_batch(imgs)
    batch_size = imgs.size(0)
    logits = logits.squeeze(1)
    outputs_diff = logits.unsqueeze(1) - logits.unsqueeze(0)  
    outputs_dist = torch.abs(outputs_diff) 
    # Flatten feature dims and compute L2 norm
    freq_features = freq_features.view(freq_features.size(0), -1) 
    freq_diff = torch.cdist(freq_features, freq_features, p=2)
    # Compute hinge loss matrix
    hinge_matrix = torch.relu(outputs_dist - tau * freq_diff)
    independence_loss = torch.sum(torch.triu(hinge_matrix, diagonal=1))

    # Normalize by number of pairs
    num_pairs = (batch_size * (batch_size - 1)) // 2
    independence_loss = independence_loss / num_pairs if num_pairs > 0 else 0.0

    return ce_loss, independence_loss