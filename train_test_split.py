import os
import shutil
import random

def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def copy_folders(src, dst, folders):
    for folder in folders:
        src_folder = os.path.join(src, folder)
        dst_folder = os.path.join(dst, folder)
        shutil.copytree(src_folder, dst_folder)

def main():
    # Define source and destination directories
    source_dir = '../data/BraTS2021'
    train_dir = '../data/train_minidata2021'
    validation_dir = '../data/validation_minidata2021'
    test_dir = '../data/test_minidata2021'

    # Create destination directories if they don't exist
    create_directory(train_dir)
    create_directory(validation_dir)
    create_directory(test_dir)

    # Get the list of all folders in the source directory
    all_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

    # Check if there are enough folders in the source directory
    if len(all_folders) < 50:
        raise ValueError("Not enough folders in the source directory for splitting into train, validation, and test sets.")

    # Shuffle the list of folders
    random.shuffle(all_folders)

    # Select folders for training, validation, and testing
    train_folders = all_folders[:30]
    validation_folders = all_folders[30:40]
    test_folders = all_folders[40:50]

    # Copy the folders to the respective destination directories
    copy_folders(source_dir, train_dir, train_folders)
    copy_folders(source_dir, validation_dir, validation_folders)
    copy_folders(source_dir, test_dir, test_folders)

    print("Folders successfully split into train, validation, and test sets.")

if __name__ == '__main__':
    main()
