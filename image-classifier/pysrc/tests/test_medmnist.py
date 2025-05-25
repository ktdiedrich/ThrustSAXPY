import medmnist
from medmnist import INFO
from medmnist import ChestMNIST
import numpy as np
import pytest
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib


matplotlib.use("Qt5Agg")


def medmnist_split_info(split, data_class, download=True):
    """
    Get the number of samples in the specified split of the ChestMNIST dataset.
    """
    DatasetClass = getattr(medmnist, data_class)
    dataset = DatasetClass(split=split, download=download)
    print(f"Dataset {split} shape:", dataset.imgs.shape)
    print(f"Labels {split} shape:", dataset.labels.shape)
    img, label = dataset[0]
    print("Sample image size:", img.size)
    print("Sample label:", label)
    import matplotlib.pyplot as plt
    plt.imshow(np.squeeze(img), cmap='gray')
    plt.title(f"Label: {label}")
    plt.show()


# @pytest.mark.parametrize("data_flag, data_class", ["chestmnist", "ChestMNIST"])
# def test_loading_medmnist(data_flag, data_class):
def test_loading_medmnist():
    """
    Test the medmnist package for the ChestMNIST dataset.
    This test checks if the dataset can be downloaded and loaded correctly.
    """
    data_flag = "chestmnist"
    data_class = "ChestMNIST"
    assert data_flag in INFO, f"INFO dictionary does not contain '{data_flag}' key."
    assert hasattr(medmnist, data_class), f"{data_class} class is not available in medmnist."
    download = True

    # Get dataset info
    info = INFO[data_flag]

    # Load the training split
    medmnist_split_info("train", data_class, download=download)
    medmnist_split_info("val", data_class, download=download)
    medmnist_split_info("test", data_class, download=download)


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