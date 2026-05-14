"""Urban ReID dataset — reads Combined_dataset CSVs directly."""

import os
import csv
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, NamedTuple

from PIL import Image
from torch.utils.data import Dataset


class Sample(NamedTuple):
    img_path: str
    pid: int       # -1 for unlabeled (competition query/test)
    camid: int
    class_label: int  # object class index (-1 if unknown)


# Fixed mapping — keeps class indices stable across runs
CLASS_TO_IDX = {
    "Container": 0,
    "Crosswalk": 1,
    "RubbishBins": 2,
    "TrafficSign": 3,
    "trafficsignal": 3,  # alias in some CSV splits
}
IDX_TO_CLASS = {
    0: "Container",
    1: "Crosswalk",
    2: "RubbishBins",
    3: "TrafficSign",
}


def _parse_camid(cam_str: str) -> int:
    """Parse 'c001' -> 1."""
    return int(cam_str.replace("c", ""))


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_split_with_pid(
    root: str, csv_path: str, classes_csv_path: Optional[str], image_dir: str
) -> List[Sample]:
    """Load a split that has objectID (train, val_query, val_test)."""
    rows = _read_csv(os.path.join(root, csv_path))

    # Load class labels if available
    class_map = {}
    if classes_csv_path:
        cls_path = os.path.join(root, classes_csv_path)
        if os.path.isfile(cls_path):
            cls_rows = _read_csv(cls_path)
            for r in cls_rows:
                key = (r["cameraID"], r["imageName"])
                cls_name = r.get("Class", "").strip()
                class_map[key] = CLASS_TO_IDX.get(cls_name, -1)

    samples = []
    for r in rows:
        cam_str = r["cameraID"]
        img_name = r["imageName"]
        pid = int(r["objectID"])
        camid = _parse_camid(cam_str)
        img_path = os.path.join(root, image_dir, img_name)
        cls_label = class_map.get((cam_str, img_name), -1)
        samples.append(Sample(img_path, pid, camid, cls_label))
    return samples


def _load_split_no_pid(
    root: str, csv_path: str, classes_csv_path: Optional[str], image_dir: str
) -> List[Sample]:
    """Load a split without objectID (competition query, test)."""
    rows = _read_csv(os.path.join(root, csv_path))

    class_map = {}
    if classes_csv_path:
        cls_path = os.path.join(root, classes_csv_path)
        if os.path.isfile(cls_path):
            cls_rows = _read_csv(cls_path)
            for r in cls_rows:
                key = (r["cameraID"], r["imageName"])
                cls_name = r.get("Class", "").strip()
                class_map[key] = CLASS_TO_IDX.get(cls_name, -1)

    samples = []
    for r in rows:
        cam_str = r["cameraID"]
        img_name = r["imageName"]
        camid = _parse_camid(cam_str)
        img_path = os.path.join(root, image_dir, img_name)
        cls_label = class_map.get((cam_str, img_name), -1)
        samples.append(Sample(img_path, -1, camid, cls_label))
    return samples


class UrbanReIDDataset:
    """Container for all splits of the Urban ReID dataset.

    Attributes:
        train: list of Sample with relabeled contiguous PIDs
        val_query / val_gallery: validation splits with original PIDs
        query / gallery: competition splits (pid=-1)
        pid_to_label: mapping from original PID to contiguous label (train only)
        num_train_pids: number of unique training identities
        num_train_cams: number of unique training cameras
    """

    def __init__(self, cfg):
        root = cfg.dataset.root

        # --- Train ---
        raw_train = _load_split_with_pid(
            root, cfg.dataset.train_csv, cfg.dataset.train_classes_csv, "image_train"
        )
        # Relabel PIDs to contiguous 0..N-1
        pid_set = sorted(set(s.pid for s in raw_train))
        self.pid_to_label = {pid: idx for idx, pid in enumerate(pid_set)}
        self.train = [
            Sample(s.img_path, self.pid_to_label[s.pid], s.camid, s.class_label)
            for s in raw_train
        ]
        self.num_train_pids = len(pid_set)
        self.num_train_cams = len(set(s.camid for s in raw_train))

        # --- Validation ---
        self.val_query = _load_split_with_pid(
            root, cfg.dataset.val_query_csv, cfg.dataset.val_query_classes_csv, "image_val_query"
        )
        self.val_gallery = _load_split_with_pid(
            root, cfg.dataset.val_test_csv, cfg.dataset.val_test_classes_csv, "image_val_test"
        )

        # --- Competition ---
        self.query = _load_split_no_pid(
            root, cfg.dataset.query_csv, cfg.dataset.query_classes_csv, "image_query"
        )
        self.gallery = _load_split_no_pid(
            root, cfg.dataset.test_csv, cfg.dataset.test_classes_csv, "image_test"
        )

        self.class_to_idx = CLASS_TO_IDX


class ImageDataset(Dataset):
    """Wraps a list of Sample with transforms for DataLoader consumption."""

    def __init__(self, samples: List[Sample], transform=None, is_train: bool = False):
        self.samples = samples
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s.img_path).convert("RGB")
        payload = {}
        if self.transform is not None:
            transformed = self.transform(img, sample=s, is_train=self.is_train)
            if isinstance(transformed, dict):
                payload.update(transformed)
            else:
                payload["images"] = transformed
        else:
            payload["images"] = img
        payload.update({
            "pids": s.pid,
            "camids": s.camid,
            "class_labels": s.class_label,
            "img_paths": s.img_path,
        })
        return payload
