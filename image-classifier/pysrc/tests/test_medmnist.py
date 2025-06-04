import medmnist
from medmnist import INFO
import numpy as np
import pytest
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib


matplotlib.use("Qt5Agg")



def plot_onehot_label_histograms(train_labels, val_labels, test_labels, class_names=None, save_path="onehot_label_histograms.png"):
    """
    Plot histograms of class indices for one-hot encoded train, val, and test labels.
    """
    label_sets = [
        ("Train Labels", train_labels),
        ("Validation Labels", val_labels),
        ("Test Labels", test_labels)
    ]
    plt.figure(figsize=(15, 4))
    for i, (title, labels) in enumerate(label_sets, 1):
        # Convert one-hot to class indices
        class_indices = np.argmax(labels, axis=1)
        plt.subplot(1, 3, i)
        plt.hist(class_indices, bins=np.arange(class_indices.max()+2)-0.5, edgecolor='black')
        plt.title(title)
        if class_names is not None:
            plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=45, ha='right')
            plt.xlabel("Class")
        else:
            plt.xlabel("Class Index")
        plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)


def plot_one_image_per_category(images, labels, class_names, save_path="category_examples.png", nrows=3):
    """
    Plot one example image for each category, labeled with class_names.
    images: numpy array of shape (N, 1, H, W) or (N, H, W)
    labels: one-hot encoded numpy array of shape (N, num_classes)
    class_names: list of class names
    """
    num_classes = len(class_names)
    ncols = int(np.ceil(num_classes / nrows))
    # Convert one-hot to class indices
    class_indices = np.argmax(labels, axis=1)
    # Find one example index for each class
    example_indices = []
    for class_idx in range(num_classes):
        idxs = np.where(class_indices == class_idx)[0]
        if len(idxs) > 0:
            example_indices.append(idxs[0])
        else:
            example_indices.append(None)  # No example for this class

    plt.figure(figsize=(2*ncols, 2*nrows))
    for i, idx in enumerate(example_indices):
        row = i // ncols
        col = i % ncols
        plt.subplot(nrows, ncols, i+1)
        if idx is not None:
            img = images[idx]
            if img.ndim == 3 and img.shape[0] == 1:
                img = img[0]
            plt.imshow(img, cmap='gray')
        plt.title(class_names[i], fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


@pytest.mark.parametrize("data_flag, data_class", [("chestmnist", "ChestMNIST")])
def test_loading_medmnist(data_flag, data_class):
    """
    This plots a histogram of the one-hot encoded labels for the train, val, and test splits.

    It also saves an example image for each category.
    :param data_flag: The flag for the dataset to be loaded (e.g., "chestmnist").
    :param data_class: The class name of the dataset in medmnist (e.g., "ChestMNIST").
    :return: None
    """
    assert data_flag in INFO, f"INFO dictionary does not contain '{data_flag}' key."
    assert hasattr(medmnist, data_class), f"{data_class} class is not available in medmnist."
    download = True

    # Get dataset info
    info = INFO[data_flag]
    labels  = info['label']
    class_names = list(labels.values())
    # Load the training split
    DataClass = getattr(medmnist, data_class)
    train_dataset = DataClass(split="train", download=download)
    val_dataset = DataClass(split="val", download=download)
    test_dataset = DataClass(split="test", download=download)

    plot_onehot_label_histograms(
        train_dataset.labels, 
        val_dataset.labels, 
        test_dataset.labels,
        class_names=class_names
    )
    plot_one_image_per_category(
        train_dataset.imgs, 
        train_dataset.labels, 
        class_names=class_names,
        save_path="category_examples_train.png",
        nrows=3
    )


def test_loading_npz():
    """
    Test the loading of NPZ files for the ChestMNIST dataset.
    This test checks if the NPZ files can be loaded correctly.
    """
    dir_path: str = "/home/ktdiedrich/data/medMNIST"
    data_flag: str = "chestmnist"
    ending: str = "npz"
    npz_path: Path  = Path(dir_path, f"{data_flag}.{ending}")
    data = np.load(npz_path)
    assert data is not None, "Failed to load NPZ file."
    train_images = data["train_images"]
    val_images = data["val_images"]
    test_images = data["test_images"]
    train_labels = data["train_labels"]
    val_labels = data["val_labels"]
    test_labels = data["test_labels"]

    assert train_images is not None, "Failed to load training data from NPZ file."
    assert val_images is not None, "Failed to load validation data from NPZ file."
    assert test_images is not None, "Failed to load test data from NPZ file."
    assert train_labels is not None, "Failed to load training labels from NPZ file."
    assert val_labels is not None, "Failed to load validation labels from NPZ file."
    assert test_labels is not None, "Failed to load test labels from NPZ file."
    assert train_images.shape[0] == train_labels.shape[0], "Mismatch in number of training images and labels."
    assert val_images.shape[0] == val_labels.shape[0], "Mismatch in number of validation images and labels."
    assert test_images.shape[0] == test_labels.shape[0], "Mismatch in number of test images and labels."
    assert train_images.shape[1:] == (28, 28), "Training images shape is incorrect."
    assert val_images.shape[1:] == (28, 28), "Validation images shape is incorrect."
    assert test_images.shape[1:] == (28, 28), "Test images shape is incorrect."
    assert train_labels.shape[1] == 14, "Training labels shape is incorrect."
    assert val_labels.shape[1] == 14, "Validation labels shape is incorrect."
    assert test_labels.shape[1] == 14, "Test labels shape is incorrect."
