import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime


########
# Data transforms
########
class permute_and_add_axis_to_mask(object):
    def __call__(self, sample):
        # Previous: (240, 240, 155, 4) , need to change to (4, 155, 240, 240) i.e. (channel, depth, height, width)
        image, mask = sample['image'], sample['mask']

        image = image.transpose((3, 2, 0, 1))
        mask = mask.transpose((2, 0, 1))

        mask= mask[np.newaxis, ...]
        return {'image':image,
                'mask':mask}
    
class spatialpad(object): # First dimension should be left untouched of [C, D, H, W]
    def __init__(self, image_target_size=[4, 256, 256, 256], mask_target_size=[1, 256, 256, 256]):
        self.image_target_size = image_target_size
        self.mask_target_size = mask_target_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask'] # image: [4, 155, 240, 240], mask[1, 155, 240, 240]

        padded_image = self.pad_input(image, self.image_target_size)

        padded_mask = self.pad_input(mask, self.mask_target_size)

        return {'image': padded_image,
                'mask': padded_mask}
    

    def pad_input(self, input_array, target_size):
        # Ensure the input array is a numpy array
        if not isinstance(input_array, np.ndarray):
            input_array = np.array(input_array)

        # Calculate padding sizes for each dimension
        pad_width = []
        for i in range(len(input_array.shape)):
            total_padding = target_size[i] - input_array.shape[i]
            if total_padding < 0:
                raise ValueError(f"Target shape must be larger than the input shape. Dimension {i} is too small.")
            pad_before = total_padding // 2
            pad_after = total_padding - pad_before
            pad_width.append((pad_before, pad_after))

        # Pad the image
        padded_image = np.pad(input_array, pad_width, mode='constant', constant_values=0)

        return padded_image   


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true): # y_pred is raw logits of shape [batch, #classes=4, slices=155, height=240, width=240], y_true is of size [batch, 1, 155, 240, 240] where each pixel value is either {0, 1, 2, 3}
        # Apply softmax to logits across channels to get probabilities
        y_pred = torch.softmax(y_pred, dim=1)

        # Convert y_true to long (integer) type if it contains float values, making it compatible for one_hot function
        if y_true.dtype != torch.long:
            y_true = y_true.long()

        # One-hot encode y_true
        batch_size, channel, *spatial_dim = y_pred.shape
        y_true_one_hot = F.one_hot(y_true, num_classes=y_pred.shape[1]).view(batch_size, channel, *spatial_dim).float() # One hot generates ([batch, channel=1, D, H, W, 4]), then reshaped to (B, C=4, D, H, W)
        assert y_true_one_hot.size() == y_pred.size()

        # Flatten the tensors
        y_pred = y_pred.contiguous().view(-1)
        y_true_one_hot = y_true_one_hot.contiguous().view(-1)

        # Calculate intersection and union
        intersection = (y_pred * y_true_one_hot).sum()
        union = y_pred.sum() + y_true_one_hot.sum()

        # Calculate Dice coefficient
        dice_coefficient = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice_coefficient # returns dice loss of a batch


class DiceScore(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predicted_class_labels, y_true): # predicted_class_labels is output from argmax of shape [batch, 1, 155, 240, 240] {0, 1, 2, 3}, and y_true is also [batch, 1, 155, 240, 240] {0, 1, 2, 3}
        '''

        args:
            predicted_clss_labels (torch.Tensor) : B1HWD
            y_true (torch.Tensor) : B1HWD
        '''
        # Get total number of classes
        num_classes = predicted_class_labels.max() + 1

        # Convert y_true to long (integer) type if it contains float values, making it compatible for one_hot function
        if y_true.dtype != torch.long:
            y_true = y_true.long()

        # Convert predicted_class_labels to long(integer) type if it contains float values, making it compatible for one_hot_function
        if predicted_class_labels.dtype != torch.long:
            predicted_class_labels = predicted_class_labels.long()

        # one-hot encode y_true
        batch_size, channel, *spatial_dim = y_true.shape # after squeeze y_true shape [batch_size, *spatial_dim]
        y_true_one_hot = F.one_hot(y_true.squeeze(), num_classes=int(num_classes)).view(batch_size, num_classes, *spatial_dim).float() # One hot of shape ([B, C=1, D, H, W, 4]), then reshaped to (B, C=4, D, H, W)

        # One-hot encode the predicted_class_labels
        predicted_one_hot = F.one_hot(predicted_class_labels.squeeze(), int(num_classes)).view(batch_size, num_classes, *spatial_dim).float()

        assert y_true_one_hot.shape == predicted_one_hot.shape

        dice_scores = []

        for cls in range(0, num_classes): # skip the background class (0)
            # Get first channel mask of predicted_one_hot and y_true_one_hot
            pred_onehot_cls = predicted_one_hot[:, cls, :, :, :] # [batch, 155, 240, 240]
            y_true_one_hot_cls = y_true_one_hot[:, cls, :, :, :] # [batch, 155, 240, 240]
            
            # Calculate dice score for the current class/channel averaged across the batch
            dice_score_cls = dice_coefficient(pred_onehot_cls, y_true_one_hot_cls)
            
            # Append the dice_score
            dice_scores.append(dice_score_cls)

        # Average dice scores across all non-background channels(classes) to get mean_dice_score
        # mean_dice_score = np.mean(dice_scores)
        
        return dice_scores
         


def dice_coefficient(y_pred, y): # takes one plane/class at a time out of 4 planes # both one-hot encoded in case we are calculating dice_score. However, when calculating DiceLoss things are different.
    smooth = 1e-10
    # y_pred = torch.round(y_pred).int()
    # y = torch.round(y).int()
            
    # Transfer tensors to CPU before converting to numpy arrays
    y_pred_np = y_pred.cpu().numpy()
    y_np = y.cpu().numpy()

    # y_pred_np = largest_connected_component(y_pred_np)

    # calculate intersection and union
    intersection = (y_pred_np * y_np).sum()
    union = y_pred_np.sum() + y_np.sum()

    # calculate dice coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth) # dice_coefficient of a class

    return dice


def save_checkpoint(model, checkpoint_dir, optimizer, epoch, train_loss, val_dice_score):
    # Generate a timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # File path to save the checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'Epoch_{epoch}_checkpoint_{timestamp}.pth')

    # Create a dictionary to save the state
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_dice_score': val_dice_score
    }

    # Save the state dictionary to a file
    torch.save(state, checkpoint_path)
    print(f'Checkpoint saved at {checkpoint_path}')


# def load_checkpoint(model, optimizer, checkpoint_path):
#     checkpoint = torch.load(checkpoint_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch = checkpoint['epoch']
#     loss = checkpoint['loss']
#     print(f"Checkpoint loaded from {checkpoint_path}")
#     return model, optimizer, epoch, loss


# To resume training from a saved checkpoint, use the following code
# Replace 'checkpoint_path' with the path to the desired checkpoint file
# model, optimizer, epoch, loss = load_checkpoint(model, optimizer, checkpoint_path)