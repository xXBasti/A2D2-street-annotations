import glob
import json
import os.path
from enum import Enum
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm


class A2D2sSet(str,Enum):
    initial_pool = "init"
    unlabeled_pool = "unlabeled_pool"
    train_set = "train"
    val_set = "val"
    test_set = "test"
    all = "all"


class A2D2s(VisionDataset):
    """`A2D2s <https://www.cs.cit.tum.de/daml/tpl/>` Dataset.

    Args:
        root (string): Root directory of dataset where directory exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "A2D2_streets_classification_labels"
    tgz_md5 = "ca58599e513873b1d001fa7d2e7013c1"
    init_pool = ["20181107_132730",
                 "20181108_091945"]
    unlabeled_pool = [
        "20181107_133258",
        "20181108_084007",
        "20180807_145028",
        "20180810_142822",
        "20180925_135056",
        "20181008_095521",
        "20181107_132300",
        "20181204_154421",
        "20181204_170238",
    ]
    val_set =[
        "20180925_101535",
        "20181016_125231",
        "20181204_135952"
    ]
    test_set =[
        "20180925_124435",
        "20181108_123750",
        "20181108_103155"
    ]
    sessions=init_pool+unlabeled_pool+val_set+test_set
    def __init__(
        self,
        root: str,
        subset: A2D2sSet = A2D2sSet.initial_pool,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        if download:
            raise NotImplementedError("Download currently not supported")
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        subset=self.select_set(subset)
        self.data, self.targets = self.load_data(subset)

    def load_data(self, subset, label_file="classification.json"):
        # now load the picked numpy arrays
        img_paths = []
        labels = []
        for folder_name in subset:
            path = os.path.join(self.root, self.base_folder,folder_name)
            session_img_paths = sorted([os.path.join(path, img_name) for img_name in
                                        list(filter(lambda p: (".jpg" in p or ".png" in p), os.listdir(path)))])
            img_paths.extend(session_img_paths)
            with open(os.path.join(path,label_file), "r") as f:
                label = json.load(f)
            labels.extend(label.values())
        return img_paths, labels

    def select_set(self, set):
        if set == A2D2sSet.initial_pool:
            return self.init_pool
        elif set == A2D2sSet.unlabeled_pool:
            return self.unlabeled_pool
        elif set == A2D2sSet.val_set:
            return self.init_pool + self.unlabeled_pool
        elif set == A2D2sSet.val_set:
            return self.val_set
        elif set == A2D2sSet.test_set:
            return self.test_set

    @staticmethod
    def resize_A2D2(dataset_folder : str, target_location : str):
        """
        Resizes the original dataset to a smaller size for classification
        Args:
            dataset_folder: original dataset folder
            target_location: flat folder structure root

        Returns:

        """
        import multiprocessing as mp
        size = (120,72)
        folders = sorted([ str(f.path).split("/")[-1] for f in os.scandir(os.path.join(dataset_folder,"camera_lidar")) if f.is_dir()])
        samples = []
        for session in folders:
            files = glob.glob(f"{dataset_folder}/camera_lidar/{session}/camera/cam_front_center/*.png")
            samples.extend(files)
        def resize_single(file):
            image = Image.open(file)
            image = Image.fromarray(np.array(image)[8:1208])
            image_re=image.resize(size)
            target_file_path=file.replace(dataset_folder,target_location).replace("/camera/cam_front_center","").replace("/camera_lidar","")
            image_re.save(target_file_path)
        with mp.Pool(20) as p:
            r = list(tqdm(p.imap(resize_single, samples), total=len(samples)))

    @staticmethod
    def setuo_A2D2(root : str, dataset_folder : str):
        """
        Args:
            root: should be the folder containing the annotations
            dataset_folder: A2D2 Bounding Box folder

        Returns:
        """
        A2D2s.resize_A2D2(dataset_folder, root)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        for filename in self.sessions:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not os.path.exists(fpath):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
