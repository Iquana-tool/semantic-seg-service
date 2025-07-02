import torch


def dice_coeff(preds: torch.Tensor, targets: torch.Tensor, epsilon=1e-6):
    preds = torch.softmax(preds, dim=1)
    preds = torch.argmax(preds, dim=1)
    preds = preds.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    intersection = (preds == targets).float().sum()
    return (2. * intersection) / (preds.numel() + targets.numel() + epsilon)


def iou_score(preds: torch.Tensor, targets: torch.Tensor, epsilon=1e-6):
    preds = torch.softmax(preds, dim=1)
    preds = torch.argmax(preds, dim=1)
    preds = preds.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    intersection = (preds == targets).float().sum()
    union = preds.numel() + targets.numel() - intersection
    return (intersection + epsilon) / (union + epsilon)
