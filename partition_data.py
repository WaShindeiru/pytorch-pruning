import os
import random
import shutil


def split_images(source_dir, train_dir, test_dir, test_ratio=0.2, max_train=None, max_test=None):
    """
    Splits images from source directory into train and test directories.

    Args:
        source_dir (str): Path to directory containing cat images
        train_dir (str): Path to train directory (will be created if doesn't exist)
        test_dir (str): Path to test directory (will be created if doesn't exist)
        test_ratio (float): Proportion of images to use for test set (default: 0.2)
    """
    # Create train and test directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get list of all image files in source directory
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Shuffle the images randomly
    random.shuffle(image_files)

    # Calculate split index
    split_idx = int(len(image_files) * test_ratio)

    # Split into test and train sets
    test_files = image_files[:split_idx]
    train_files = image_files[split_idx:]

    if max_test is not None:
        test_files = test_files[:max_test]
    if max_train is not None:
        train_files = train_files[:max_train]

    # Copy files to test directory
    for file in test_files:
        src_path = os.path.join(source_dir, file)
        dst_path = os.path.join(test_dir, file)
        shutil.copy2(src_path, dst_path)

    # Copy files to train directory
    for file in train_files:
        src_path = os.path.join(source_dir, file)
        dst_path = os.path.join(train_dir, file)
        shutil.copy2(src_path, dst_path)

    print(f"Split complete: {len(train_files)} images in train, {len(test_files)} images in test")


# Example usage
if __name__ == "__main__":
    # Set your directory paths here
    source_directory = "/home/washindeiru/studia/sem_8/ssn/sem/pytorch-pruning/data/kagglecatsanddogs_5340/PetImages/Cat"
    train_directory = "/home/washindeiru/studia/sem_8/ssn/sem/pytorch-pruning/data/kagglecatsanddogs_5340/Train/cats"
    test_directory = "/home/washindeiru/studia/sem_8/ssn/sem/pytorch-pruning/data/kagglecatsanddogs_5340/Test/cats"

    split_images(source_directory, train_directory, test_directory, 0.15, 1200, 400)