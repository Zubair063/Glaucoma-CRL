import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
try:
    from transformers import AutoTokenizer
    USE_TRANSFORMERS = True
except ImportError:
    from pytorch_pretrained_bert import BertTokenizer
    USE_TRANSFORMERS = False


class FairCLIPDataset(Dataset):
    """
    FairCLIP Dataset Loader
    Loads images from .npz files and text from CSV file
    """
    
    def __init__(self, data_dir, csv_path, split='training', transform=None, max_seq_len=512):
        """
        Args:
            data_dir: Directory containing Training/, Testing/, Validation/ folders with .npz files
            csv_path: Path to data_summary.csv
            split: 'training', 'test', or 'validation'
            transform: Image transforms
            max_seq_len: Maximum sequence length for text
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.max_seq_len = max_seq_len
        
        # Load CSV data
        self.df = pd.read_csv(csv_path)
        
        # Filter by split - match the directory structure
        # Map split names: 'training' -> 'Training', 'validation' -> 'Validation', 'test' -> 'Testing'
        split_to_dir_map = {
            'training': 'Training',
            'validation': 'Validation',
            'test': 'Testing'
        }
        split_dir_name = split_to_dir_map.get(split, split.capitalize())
        
        # Filter CSV by split
        self.df = self.df[self.df['use'] == split].reset_index(drop=True)
        
        # Get split directory
        split_dir = os.path.join(data_dir, split_dir_name)
        self.split_dir = split_dir
        
        # Get list of actual files in the directory
        if os.path.exists(split_dir):
            available_files = set(os.listdir(split_dir))
            # Filter dataframe to only include files that exist
            self.df = self.df[self.df['filename'].isin(available_files)].reset_index(drop=True)
        
        # Initialize AutoTokenizer (same as MultiFair)
        if USE_TRANSFORMERS:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            except:
                self.tokenizer = AutoTokenizer.from_pretrained('./bert-base-uncased')
        else:
            try:
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            except:
                self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased', do_lower_case=True)
        
        print(f"Loaded {len(self.df)} samples from {split} split")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get filename and label
        filename = self.df.iloc[idx]['filename']
        label = self.df.iloc[idx]['glaucoma']
        
        # Try to get text from 'note' field first (as in MultiFair), fallback to 'gpt4_summary'
        if 'note' in self.df.columns:
            text = self.df.iloc[idx]['note']
            if pd.isna(text) or str(text).strip() == '':
                text = self.df.iloc[idx].get('gpt4_summary', 'no clinical notes available')
        else:
            text = self.df.iloc[idx].get('gpt4_summary', 'no clinical notes available')
        
        # Convert text to string and handle edge cases
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        elif not isinstance(text, str):
            text = str(text)
        
        # Clean text (convert to lowercase like MultiFair)
        text = text.lower().strip()
        if len(text) < 10:
            text = "no clinical notes available"
        
        # Convert label to binary
        if isinstance(label, str):
            label = 1 if label.lower() == 'yes' else 0
        else:
            label = int(label)
        
        # Load image from npz file
        npz_path = os.path.join(self.split_dir, filename)
        data = np.load(npz_path, allow_pickle=True)
        image = data['slo_fundus']  # Shape: (664, 512)
        
        # Convert to PIL Image and apply transforms
        image = Image.fromarray(image.astype(np.uint8)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform
            image = transforms.ToTensor()(image)
        
        # Tokenize text using AutoTokenizer (same as MultiFair)
        if USE_TRANSFORMERS:
            encoded = self.tokenizer(
                text,
                max_length=self.max_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            txt_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
            segment_ids = torch.zeros_like(txt_ids)
        else:
            encoded = self.tokenizer(
                text,
                max_length=self.max_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            txt_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
            segment_ids = torch.zeros_like(txt_ids)
        
        return {
            'image': image,
            'txt': txt_ids,
            'mask': attention_mask,
            'segment': segment_ids,
            'label': torch.tensor(label, dtype=torch.long),
            'filename': filename
        }


def get_fairclip_loaders(data_dir, csv_path, batch_size=16, num_workers=4, max_seq_len=512):
    """
    Get data loaders for FairCLIP dataset
    """
    # Image transforms - more augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33))
    ])
    
    # Simple transform for validation and test (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create datasets
    train_dataset = FairCLIPDataset(data_dir, csv_path, split='training', 
                                    transform=train_transform, max_seq_len=max_seq_len)
    val_dataset = FairCLIPDataset(data_dir, csv_path, split='validation', 
                                  transform=eval_transform, max_seq_len=max_seq_len)
    test_dataset = FairCLIPDataset(data_dir, csv_path, split='test', 
                                   transform=eval_transform, max_seq_len=max_seq_len)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

