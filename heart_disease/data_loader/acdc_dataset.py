import os
import sys
from sqlalchemy import label
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from .preprocessing import preprocess_volume


"""class ACDCDataset(Dataset):
    def __init__(self, root_dir, split="training"):
        self.root_dir = os.path.join(root_dir, split)

        self.patient_dirs = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])

    def _read_label(self, patient_path):
        info_path = os.path.join(patient_path, "Info.cfg")
        
        with open(info_path, "r") as f:
            lines = f.readlines()
        
        for line in lines:
            if "Group" in line:
                label = line.split(":")[1].strip()
                break

        label_map = {
            "NOR": 0,
            "DCM": 1,
            "HCM": 2,
            "MINF": 3,
            "RV": 4
        }

        return label_map[label]

    def _load_cine_volume(self, patient_path):
        nii_files = sorted([
        f for f in os.listdir(patient_path)
            if f.endswith(".nii.gz") and "_gt" not in f
        ])

        path = os.path.join(patient_path, nii_files[0])
        img = nib.load(path).get_fdata()

        return img
        

    def _prepare_samples(self):
        samples = []

        for patient in self.patient_dirs:
            patient_path = os.path.join(self.root_dir, patient)

            cine = self._load_cine_volume(patient_path)
            label = self._read_label(patient_path)

            samples.append((cine, label))

        return samples

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient = self.patient_dirs[idx]
        patient_path = os.path.join(self.root_dir, patient)

        cine = self._load_cine_volume(patient_path)
        label = self._read_label(patient_path)

        processed = preprocess_volume(cine)

        return torch.tensor(processed, dtype=torch.float32), torch.tensor(label)
"""

#acdc_dataset.py
import os
import nibabel as nib
import torch
import torchio as tio
from torch.utils.data import Dataset

class ACDCDataset(Dataset):
    def __init__(self, data_dir, transform=None, stage='train'):
        self.data_dir = data_dir
        self.stage = stage
        self.label_map = {'NOR': 0, 'MINF': 1, 'DCM': 2, 'HCM': 3, 'RVA': 4}

        import glob
        search_pattern = os.path.join(self.data_dir, "**/*.nii.gz")
        all_files = glob.glob(search_pattern, recursive=True)
        self.subjects = [f for f in all_files if "_gt" not in f]
        self.subjects.sort()

        
        common_transforms = [
            tio.RescaleIntensity(out_min_max=(0, 1)),
            tio.ZNormalization(),
            
            tio.CropOrPad((224, 224, 1)),
        ]

        if transform and stage == 'train':
            self.transform = tio.Compose([
                *common_transforms,
                tio.RandomAffine(scales=(0.9, 1.1), degrees=15),
                tio.RandomElasticDeformation(
                    num_control_points=4,
                    max_displacement=2,
                    locked_borders=0
                ),
            ])
        else:
            self.transform = tio.Compose(common_transforms)

    def _get_label_from_path(self, path):
       
        
        parent_dir = os.path.dirname(path)
        info_file = os.path.join(parent_dir, "Info.cfg")

        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                content = f.read()
                for key in self.label_map.keys():
                    if key in content:
                        return self.label_map[key]

        
        return 0

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        path = self.subjects[idx]
        img_nifti = nib.load(path)
        img = img_nifti.get_fdata()

        
        if img.ndim == 4:
            img = img[:, :, :, 0]

        
        tensor = torch.from_numpy(img).float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        subject = tio.Subject(image=tio.ScalarImage(tensor=tensor))

        
        transformed = self.transform(subject)
        image_tensor = transformed.image.data 

       
        image_tensor = image_tensor.squeeze(-1) 
        image_tensor = image_tensor.unsqueeze(0) 

        label = self._get_label_from_path(path)

        return image_tensor, torch.tensor(label).long()

