#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[110]:


import os
import torch
from torch.utils.data import Dataset, DataLoader

from glob import glob as glob
import nibabel as nib
import numpy as np
import datetime
import random

import monai
from monai.networks.nets import UNet
from monai.transforms import Compose
from monai.visualize import plot_2d_or_3d_image
from monai.metrics import CumulativeIterationMetric
from monai.losses import DiceLoss
from monai.metrics.meandice import DiceMetric

from utilities import save_checkpoint
from utilities import permute_and_add_axis_to_mask, spatialpad

import csv


# In[111]:


my_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
my_device


# ## Dataset

# In[112]:


main_directory = "../data/train_minidata2021"
output_directory = "../data/train_minidata2021/processed_train"
main_directory_val = "../data/validation_minidata2021"
output_directory_val = "../data/validation_minidata2021/processed_val"


# In[113]:


def load_nifty(directory, example_id, suffix):
    file_path = os.path.join(directory, example_id + "_" + suffix + ".nii.gz")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file or no access: '{file_path}'")
    return nib.load(file_path)


# In[114]:


def load_channels(directory, example_id):
    flair = load_nifty(directory, example_id, "flair")
    t1 = load_nifty(directory, example_id, "t1")
    t1ce = load_nifty(directory, example_id, "t1ce")
    t2 = load_nifty(directory, example_id, "t2")
    return flair, t1, t1ce, t2


# In[115]:


def prepare_nifty(d, output_dir):
    example_id = d.split("/")[-1]
    flair, t1, t1ce, t2 = load_channels(d, example_id)
    affine, header = flair.affine, flair.header

    # Load data for each modality
    flair_data = get_data(flair)
    t1_data = get_data(t1)
    t1ce_data = get_data(t1ce)
    t2_data = get_data(t2)

    # Combine into a single volume
    combined_volume = np.stack([flair_data, t1_data, t1ce_data, t2_data], axis=3)

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the combined volume in the output directory
    img_path = os.path.join(output_dir, example_id + ".nii.gz")
    vol = nib.Nifti1Image(combined_volume, affine, header=header)
    nib.save(vol, img_path)

    # Process segmentation if available
    mask_path = ""
    seg_path = os.path.join(d, example_id + "_seg.nii.gz")
    if os.path.exists(seg_path):
        seg = nib.load(seg_path)
        affine, header = seg.affine, seg.header
        seg_data = get_data(seg, "uint8")
        seg_data[seg_data == 4] = 3  # Adjust labels if necessary
        seg = nib.Nifti1Image(seg_data, affine, header=header)
        mask_path = os.path.join(output_dir, example_id + "_seg.nii.gz")
        nib.save(seg, mask_path)

    return img_path, mask_path


# In[116]:


def get_data(nifty, dtype="int16"):
    if dtype == "int16":
        data = np.abs(nifty.get_fdata().astype(np.int16))
        data[data == -32768] = 0  # Handle edge cases
        return data
    return nifty.get_fdata().astype(np.uint8)
# Initialize lists to store paths
train_volume_path = []
train_segmentation_path = []


# In[117]:


for subject_dir in os.listdir(main_directory):
    subject_path = os.path.join(main_directory, subject_dir)
    if os.path.isdir(subject_path):  # Check if it's a directory
        print(f"Processing {subject_path}...")
        try:
            # List the contents of the directory
            files = os.listdir(subject_path)
            print(f"Files in {subject_path}: {files}")
            
            img_path, mask_path = prepare_nifty(subject_path, output_directory)
            train_volume_path.append(img_path)
            train_segmentation_path.append(mask_path if mask_path else None)
        except FileNotFoundError as e:
            print(e)  # Print the error if any file is missing
        except Exception as e:
            print(f"An error occurred while processing {subject_path}: {e}")

# Convert lists to arrays
train_volume_path = np.array(train_volume_path)
train_segmentation_path = np.array(train_segmentation_path)

# Print the arrays
print("Train Volume Paths:", train_volume_path)
print("Train Segmentation Paths:", train_segmentation_path)


# In[118]:


train_volumes_path = train_volume_path
train_segmentations_path = train_segmentation_path


# In[119]:


train_volumes_path


# In[120]:


train_segmentations_path


# In[121]:


val_volumes_path = []
val_segmentations_path = []


# In[122]:


for subject_dir in os.listdir(main_directory_val):
    subject_path = os.path.join(main_directory_val, subject_dir)
    if os.path.isdir(subject_path):  # Check if it's a directory
        print(f"Processing {subject_path}...")
        try:
            # List the contents of the directory
            files = os.listdir(subject_path)
            print(f"Files in {subject_path}: {files}")
            
            img_path, mask_path = prepare_nifty(subject_path, output_directory)
            val_volumes_path.append(img_path)
            val_segmentations_path.append(mask_path if mask_path else None)
        except FileNotFoundError as e:
            print(e)  # Print the error if any file is missing
        except Exception as e:
            print(f"An error occurred while processing {subject_path}: {e}")

# Convert lists to arrays
train_volume_path = np.array(val_volumes_path)
train_segmentation_path = np.array(val_segmentations_path)

# Print the arrays
print("val Volume Paths:", train_volume_path)
print("val Segmentation Paths:", train_segmentation_path)


# In[123]:


val_segmentations_path


# In[124]:


val_volumes_path


# In[125]:


list(zip(train_volumes_path,train_segmentations_path))[:2]


# In[ ]:





# In[126]:


batch_size=2


# In[127]:


class BratsDataset(Dataset):
    def __init__(self, images_path_list, masks_path_list, transform=None):
        """
        Args:
            images_path_list (list of strings): List of paths to input images.
            masks_path_list (list of strings): List of paths to masks.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_path_list = images_path_list
        self.masks_path_list = masks_path_list
        self.transform = transform
        self.length = len(images_path_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Load image
        image_path = self.images_path_list[idx]
        image = nib.load(image_path).get_fdata()
        image = np.float32(image) # shape of image [240, 240, 155, 4]

        # Load mask
        mask_path = self.masks_path_list[idx]
        mask = nib.load(mask_path).get_fdata()
        mask = np.float32(mask) # shape of mask [240, 240, 155]

        if self.transform:
            transformed_sample = self.transform({'image': image, 'mask': mask})
        
        return transformed_sample


# In[ ]:





# ## Data Transform

# In[128]:


data_transform = Compose([ # input image of shape [240, 240, 155, 4]
    permute_and_add_axis_to_mask(), # image: [4, 155, 240, 240], mask[1, 155, 240, 240] # new channel in the first dimension is added in mask inorder to make compatible with Resize() as Resize takes only 4D tensor
    spatialpad(image_target_size=[4, 256, 256, 256], mask_target_size=[1, 256, 256, 256]),
])


# In[129]:


train_ds = BratsDataset(
    train_volumes_path,
    train_segmentations_path,
    transform=data_transform
)

val_ds = BratsDataset(
    val_volumes_path,
    val_segmentations_path,
    transform=data_transform
)


# ## DataLoader

# In[130]:


# Create dataloaders
train_loader = DataLoader(dataset=train_ds,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True)
val_loader = DataLoader(dataset=val_ds,
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=True)


# ## Model

# In[131]:


# Instantiate a U-Net model
model = UNet(
    spatial_dims=3,        # 3 for using 3D ConvNet and 3D Maxpooling
    in_channels=4,         # since 4 modalities
    out_channels=4,        # 4 sub-regions to segment
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
).to(my_device)
print(model)


# # Training

# In[132]:


def train_step(model,
               dataloader,
               loss_fn,
               dice_score,
               optimizer,
               batch_train_loss_csv,
               epoch_num):
               
    # Putting the model in train mode
    model.train()

    # Initialize train_loss list
    train_loss = []

    # Loop through batches of data
    for batch_num, batch_data in enumerate(dataloader):
        X = batch_data['image'] # torch.Size([batch, 4, 128, 240, 240])
        Y = batch_data['mask'] # torch.Size([batch, 1, 128, 240, 240]) (batch, 1, 128, 240, 240) (multi-class i.e. a pixel_value ~ {0, 1, 2, 3})

        # Send data to target device
        X, Y = X.to(my_device), Y.to(my_device)

        optimizer.zero_grad() # Clear previous gradients

        # Forward pass
        y_pred = model(X) # y_pred shape torch.Size([batch, 4, 128, 240, 240]) # produces raw logits. 4 is due to 4 sub regions, not 4 modalities.
        
        # Compute and accumulate loss
        loss = loss_fn(y_pred, Y) # loss one-hot encodes the y so y will be [batch, 4, 128, 240, 240] and y_pred is [batch, 4, 128, 240, 240], loss is scalar (may be averages across modalities and batch as well)

        # Backpropagation and Optimization
        loss.backward() # Compute gradients
        optimizer.step() # Update weights
        tr_loss = loss.item()

        # Accumulate train_loss for log
        train_loss.append(tr_loss)

        with torch.no_grad():
            # Calculate and accumulate metric across the batch
            predicted_class_labels = torch.argmax(y_pred, dim=1, keepdim=True) # After argmax with keepdim=True: [batch, 1, D, H, W] {0, 1, 2, 3}, since it takes argmax along the channels(or, the #classes)

        # Write batch wise train loss to csv file
        with open(batch_train_loss_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch_num+1, batch_num+1, tr_loss])

        print(f'Iteration: {batch_num + 1} ---|---  Loss {tr_loss:.3f}')

    # Average the dice loss (i.e. average of batches ~ 1 epoch)
    train_loss_average = np.mean(train_loss) # for checkpoint

    return  train_loss_average


# In[133]:


def val_step(model,
              dataloader,
              dice_score:CumulativeIterationMetric):

    # Putting model in eval mode
    model.eval()

    # Initialize validation dice score
    val_dice_score = 0

    dice_score.reset()

    # Turn on inference context manager
    with torch.inference_mode(): # Disable gradient computation 

        # Loop through batches of data in dataloader
        for batch_num, batch_data in enumerate(dataloader):
            X = batch_data['image'] # [batch, 4, D, H, W]
            Y = batch_data['mask'] # [B, 1, D, H, W]

            # Send data to target device
            X, Y = X.to(my_device), Y.to(my_device)

            # Forward pass
            test_pred_logits = model(X) # [B, 4, 128, 240, 240]

            # Calculate and accumulate metric across the batch
            predicted_class_labels = torch.argmax(test_pred_logits, dim=1, keepdim=True) # test_pred_logits of shape [batch, 4, D, H, W] {raw logits}, after argmax [batch, D, H, W] {0, 1, 2, 3}, since it takes argmax along the channels(or, the #classes)
            batch_dice_score = dice_score(predicted_class_labels, Y)
            
            # print(f"DSC (batch wise): {batch_dice_score}")

    # Aggregate dice score (epoch wise)
    val_dice_score = dice_score.aggregate().item()

    return val_dice_score


# In[134]:


from tqdm.auto import tqdm

# Various parameters required for training and test step
def train(model,
          checkpoint_dir,
          batch_train_loss_csv,
          epoch_val_dsc_csv,
          train_loader,
          val_loader,
          optimizer,
          loss_fn,
          dice_score,
          epochs):
    
    # # Creating empty list to hold loss and dice_score
    # results = {
    #     'train_loss':[], # This store epoch wise dice loss
    #     'val_dice_score':[] # This stores epoch wise validation dice score
    # }

    # Looping through traininig and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        print(f'----------- Epoch: {epoch+1} ----------- \n')

        train_loss = train_step(model=model,
                                      dataloader=train_loader,
                                      loss_fn=loss_fn,
                                      dice_score=dice_score,
                                      optimizer=optimizer,
                                      batch_train_loss_csv=batch_train_loss_csv,
                                      epoch_num=epoch)
        
        val_dice_score = val_step(model=model,
                                  dataloader=val_loader,
                                  dice_score=dice_score)

        # Write val_dice_score to csv file
        with open(epoch_val_dsc_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, val_dice_score])

        print(f'--|-- Epoch {epoch+1} Validation DS: {val_dice_score:.4f} --|--')

        # Save checkpoint
        save_checkpoint(model, checkpoint_dir, optimizer, epoch+1, np.mean(train_loss), val_dice_score)

        # # Append to the list
        # results['train_loss'].append(np.mean(train_loss))
        # results['val_dice_score'].append(val_dice_score)

    # return results


# In[135]:


# Model name
model_name = '3DUNet'

# Generate a timestamp
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M') # -%S

# path directory to save the checkpoints and logs
log_dir = os.path.join('logs_and_checkpoints', 'logs',model_name, timestamp)
checkpoint_dir = os.path.join('logs_and_checkpoints', 'checkpoints', model_name, timestamp)

# Create the directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

print(f"Checkpoints will be saved in: {log_dir}")
print(f"Logs will be saved in: {checkpoint_dir}")


# In[136]:


# Set random seeds
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
monai.utils.set_determinism(seed=random_seed)

# Set the number of epochs, loss function and optimizer
num_epochs = 500
dice_loss = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
dice_score = DiceMetric(include_background=False)

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)


# Create files for logs
# batch_train_loss_csv to save batch wise loss
# epoch_val_dsc_csv to save epoch wise loss
batch_train_loss_csv = os.path.join(log_dir, f'batch_train_loss_{log_dir[-16:]}.csv')
epoch_val_dsc_csv = os.path.join(log_dir, f'epoch_val_dsc_{log_dir[-16:]}.csv')

with open(batch_train_loss_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Iteration', 'Train Loss'])

with open(epoch_val_dsc_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Val DSC'])


# Start the timer
from timeit import default_timer as timer
start_time = timer()

# Train model
model_results = train(model,
                      checkpoint_dir,
                      batch_train_loss_csv,
                      epoch_val_dsc_csv,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      optimizer=optimizer,
                      loss_fn=dice_loss,
                      dice_score=dice_score,
                      epochs=num_epochs)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")


# In[ ]:




