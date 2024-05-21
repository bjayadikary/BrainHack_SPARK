#!/usr/bin/env python
# coding: utf-8

# In[93]:


import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor

from glob import glob as glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

import random
#import cv2

from torchsummary import summary
from skimage import transform
from skimage.transform import resize
import torchio as tio
import wandb


# In[94]:


my_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
my_device


# In[95]:


wandb.login()


# In[96]:


wandb.init(
    project="BrainTumor2021 Implementation",
    config={
        "epochs":5,
        "batch_size":2,
        "lr":0.001
    }
)


# # Dataset

# In[97]:


main_directory = "../data/train_minidata2021"
output_directory = "../data/train_minidata2021/processed_train"
main_directory_val = "../data/validation_minidata2021"
output_directory_val = "../data/validation_minidata2021/processed_val"


# In[98]:


def load_nifty(directory, example_id, suffix):
    file_path = os.path.join(directory, example_id + "_" + suffix + ".nii.gz")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file or no access: '{file_path}'")
    return nib.load(file_path)


# In[99]:


def load_channels(directory, example_id):
    flair = load_nifty(directory, example_id, "flair")
    t1 = load_nifty(directory, example_id, "t1")
    t1ce = load_nifty(directory, example_id, "t1ce")
    t2 = load_nifty(directory, example_id, "t2")
    return flair, t1, t1ce, t2


# In[100]:


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


# In[101]:


def get_data(nifty, dtype="int16"):
    if dtype == "int16":
        data = np.abs(nifty.get_fdata().astype(np.int16))
        data[data == -32768] = 0  # Handle edge cases
        return data
    return nifty.get_fdata().astype(np.uint8)
# Initialize lists to store paths
train_volume_path = []
train_segmentation_path = []


# In[102]:


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


# In[103]:


def get_data(nifty, dtype="int16"):
    if dtype == "int16":
        data = np.abs(nifty.get_fdata().astype(np.int16))
        data[data == -32768] = 0  # Handle edge cases
        return data
    return nifty.get_fdata().astype(np.uint8)

# Initialize lists to store paths
val_volumes_path = []
val_segmentations_path = []


# In[104]:


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
val_volumes_path = np.array(val_volumes_path)
val_segmentations_path = np.array(val_segmentations_path)

# Print the arrays
print("val Volume Paths:", val_volumes_path)
print("val Segmentation Paths:", val_segmentations_path)


# In[105]:


nib.load(train_volume_path[0]).get_fdata().shape


# In[106]:


nib.load(train_segmentation_path[0]).get_fdata().shape


# In[107]:


nib.load(val_volumes_path[0]).get_fdata().shape


# In[108]:


nib.load(val_segmentations_path[0]).get_fdata().shape


# In[109]:


class permute_and_add_axis_to_mask(object):
    def __call__(self, sample):
        # Previous: (240, 240, 155, 4) , need to change to (4, 155, 240, 240) i.e. (channel, depth, height, width)
        image, mask = sample['image'], sample['mask']

        image = image.transpose((3, 2, 0, 1))
        mask = mask.transpose((2, 0, 1))

        mask= mask[np.newaxis, ...]
        return {'image':image,
                'mask':mask}


# In[110]:


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

        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)
        
        return sample


# In[111]:


rescale_transform = tio.Resize(
    keys=['image', 'mask'],
    target_shape=(128, 240, 240)
)


# In[112]:


data_transform = Compose([ # input image of shape [240, 240, 155, 4]
    permute_and_add_axis_to_mask(), # image: [4, 155, 240, 240], mask[1, 155, 240, 240] # new channel in the first dimension is added in mask inorder to make compatible with Resize() as Resize takes only 4D tensor
    rescale_transform,
])


# In[113]:


train_ds = BratsDataset(
    train_volume_path,
    train_segmentation_path,
    transform=data_transform
)

val_ds = BratsDataset(
    val_volumes_path,
    val_segmentations_path,
    transform=data_transform
)


# In[114]:


train_ds[0]['image'].shape # previously numpy array of (240, 240, 155, 4), Now changed to: (4, 155, 240, 240)


# In[115]:


train_ds[0]['mask'].shape # Before: (240, 240, 155) After:(1, 128, 240, 240)


# In[116]:


sample_patient = train_ds[1]
sample_patient['image'].shape


# In[117]:


# This was made for previous dimension, now the dimension is changed, so need resurrection
# modality = 1
# slice_idx = 75
# plt.imshow(sample_patient['image'][:,:, slice_idx, modality], cmap='gray')
# plt.imshow(sample_patient['mask'][:,:, slice_idx], alpha=0.4)
# plt.show()

modality = 0
slice_idx = 90
plt.imshow(sample_patient['image'][modality, slice_idx, :, :], cmap='gray')
plt.imshow(sample_patient['mask'][0, slice_idx, :, :], alpha=0.4) # since channel dimension is also added on the mask volume
plt.show()


# In[ ]:





# In[118]:


# patient_num = random.randint(0, len(train_ds)-1) # random patient
# slice_num = 75 # Extract 75th slice

# sample_image = train_ds[patient_num]['image'] # (240, 240, 155, 4)
# sample_mask = train_ds[patient_num]['mask'] # (240, 240, 155)

# sample_image_slice = sample_image[:,:, slice_num, :] # (240, 240, 4)
# sample_mask_slice = sample_mask[:,:, slice_num] # (240, 240)

# fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 15))
# labels = []

# # Iterate through first 4 modalities and display them
# for i, ax in enumerate(axes[:-1]):
#     ax.set_xlabel(f"")
#     ax.imshow(sample_image_slice[:,:,i], cmap='gray') # prints 4 modalities of image in 4 columns
#     ax.set_xticks([])
#     ax.set_yticks([])

# # Display the mask in last column
# axes[-1].imshow(sample_mask_slice)
# axes[-1].set_xticks([])
# axes[-1].set_yticks([])
# plt.show()


# In[119]:


patient_num = random.randint(0, len(train_ds)-1) # random patient
slice_idx = 75 # Extract 75th slice

sample_image = train_ds[patient_num]['image'] # (4, 128, 240, 240)
sample_mask = train_ds[patient_num]['mask'] # (1, 128, 240, 240)
sample_image_slice = sample_image[:, slice_idx, :, :] # (4, 240, 240)
sample_mask_slice = sample_mask[0,slice_idx,:, :] # (240, 240)
print('sample_mask_slice', sample_mask_slice.shape)
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 15))
labels = []

# Iterate through first 4 modalities and display them
for i, ax in enumerate(axes[:-1]):
    ax.set_xlabel(f"")
    ax.imshow(sample_image_slice[i,:,:], cmap='gray') # prints 4 modalities of image in 4 columns
    ax.set_xticks([])
    ax.set_yticks([])

# Display the mask in last column
axes[-1].imshow(sample_mask_slice)
axes[-1].set_xticks([])
axes[-1].set_yticks([])
plt.show()


# # DataLoader

# In[120]:


# Create dataloader
train_loader = DataLoader(dataset=train_ds,
                          batch_size=1,
                          shuffle=True)
val_loader = DataLoader(dataset=val_ds,
                        batch_size=1,
                        shuffle=False)


# Sanity Check: Iterate through dataloader and check the size

# In[121]:


for batch in train_loader:
    images = batch['image']
    masks = batch['mask']
    print(f'Batch images shape {images.shape}')
    print(f'Batch masks shape {masks.shape}')
    print('--------------------')
    break


# In[122]:


len(train_loader) # Get the total number of batches. Returned 5, which means 5 batch of data each with shape(batch_size=2, 4, 155, 240, 240)


# In[123]:


a_batch_image_mask = next(iter(train_loader))


# In[124]:


a_batch_mask = a_batch_image_mask['mask']


# In[125]:


a_batch_mask.shape


# In[126]:


a_mask_volume = a_batch_mask[0, :, :, :] # select first sample from the batch
a_mask_volume.shape


# In[127]:


a_mask_slice = a_mask_volume[0, 75, :, :] # get 75th slice
a_mask_slice.shape


# # Model

# In[128]:


import torch
import monai


# In[129]:


from monai.networks.nets import UNet
from monai.losses import DiceLoss


# In[130]:


# Instantiate a U-Net model
model = UNet(
    spatial_dims=3,        # 3 for using 3D ConvNet and 3D Maxpooling
    in_channels=4,         # since 4 modalities
    out_channels=4,        # 4 sub-regions to segment
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
).to(my_device)
print(model)


# In[131]:


summary(model, input_size=(4, 128, 240, 240))  # channels, depth, height, width


# # Training

# In[132]:


def train_step(model,
               dataloader,
               loss_fn,
               optimizer):
    # Putting the model in train mode
    model.train()

    # Initialize train loss
    train_loss = 0

    # Loop through batches of data
    for batch_num, batch_data in enumerate(dataloader):
        X = batch_data['image']
        Y = batch_data['mask'] # torch.Size([1, 1, 128, 240, 240]) (batch, channel=1, 128, 240, 240)

        # Send data to target device
        X, Y = X.to(my_device), Y.to(my_device)

        # Forward pass
        y_pred = model(X) # y_pred shape torch.Size([batch, 4, 128, 240, 240])
        
        # Compute and accumulate loss
        loss = loss_fn(y_pred, Y) # loss one-hot encode the y so y will be [batch, 4, 128, 240, 240] and y_pred is [batch, 4, 128, 240, 240], thus loss is scalar(loss across batch)
        train_loss += loss.item()

        # Backpropagation and Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate and accumulate metric across the batch
        # y_pred_class = torch.argmax(y_pred, dim=1) # y_pred_class shape [batch, 1, 128, 240, 240] since it takes argmax the channels

    # Average loss and accuracy of all batches (average of all batches = 1 epoch)
    train_loss = train_loss / len(dataloader)

    return train_loss


# In[133]:


def val_step(model,
              dataloader,
              loss_fn):
    # Putting model in eval mode
    model.eval()

    # Initialize test loss
    val_loss = 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through batches of data in dataloader
        for batch_num, batch_data in enumerate(dataloader):
            X = batch_data['image']
            Y = batch_data['mask']

            # Send data to target device
            X, Y = X.to(my_device), Y.to(my_device)

            # Forward pass
            test_pred_logits = model(X)

            # Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, Y)
            val_loss += loss.item()

    return val_loss


# In[134]:


from tqdm.auto import tqdm

# Various parameters required for training and test step
def train(model,
          train_loader,
          val_loader,
          optimizer,
          loss_fn,
          epochs):
    
    # Creating empty list to hold loss and accuracy
    results = {
        'train_loss':[],
        'val_loss':[],
    }

    # Looping through traininig and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss= train_step(model=model,
                                dataloader=train_loader,
                                loss_fn=loss_fn,
                                optimizer=optimizer)
        
        val_loss = val_step(model=model,
                              dataloader=val_loader,
                              loss_fn=loss_fn)
        
        # Print and append the loss current epoch
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss
        })
        print(
            f'Epoch: {epoch+1} |'
            f'train_loss: {train_loss:.4f} |'
            f'val_loss: {val_loss:.4f} |'

        )

        # Append to the list
        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)

    return results


# In[135]:


# Set random seeds
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
monai.utils.set_determinism(seed=random_seed)

# Set the number of epochs, loss function and optimizer
num_epochs = 1
loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)


# Start the timer
from timeit import default_timer as timer
start_time = timer()

# Train model
model_results = train(model,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      epochs=num_epochs)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




