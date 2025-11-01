#!/usr/bin/env python3
"""Preprocess HCP parcellated data into expected format for Brain-JEPA"""
import os
import json
import glob
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_label_mapping(json_path):
    """Load subject ID to label mapping"""
    with open(json_path, 'r') as f:
        mapping = json.load(f)
    # Keep as string keys (matching meta['sub'] format) and convert values to int
    return {str(k): int(v) for k, v in mapping.items()}

def process_pt_file(filepath):
    """Load and process a single .pt file"""
    data = torch.load(filepath, map_location='cpu')
    
    # Extract subject ID from meta (it's stored as a string)
    subject_id = str(data['meta']['sub'])
    
    # Extract bold signal [n_frames, 450]
    bold = data['bold']  # Shape: [n_frames, 450]
    
    # Transpose to [450, n_frames] to match expected format (ROIs x timepoints)
    bold = bold.T  # Now shape: [450, n_frames]
    
    # Pad or interpolate to seq_length=490
    n_frames = bold.shape[1]
    seq_length = 490
    
    if n_frames < seq_length:
        # Pad with last value
        padding = bold[:, -1:].repeat(1, seq_length - n_frames)
        bold = torch.cat([bold, padding], dim=1)
    elif n_frames > seq_length:
        # Downsample/interpolate to 490
        indices = torch.linspace(0, n_frames - 1, seq_length).long()
        bold = bold[:, indices]
    
    return subject_id, bold.float()

def preprocess_hcp_data(
    data_dir='data',
    label_map_path='hcp_sex_target_id_map.json',
    output_dir='brain-jepa-dataset',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    """Preprocess HCP data into expected format"""
    
    # Validate split ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
    print(f"Loading label mapping from {label_map_path}...")
    label_mapping = load_label_mapping(label_map_path)
    print(f"Loaded {len(label_mapping)} subject labels")
    
    # Find all .pt files
    print(f"\nScanning for .pt files in {data_dir}...")
    pt_files = []
    
    # Try multiple patterns to find files
    patterns = [
        f'{data_dir}/hcp-parc_*/*.pt',  # Direct in hcp-parc_* directories
        f'{data_dir}/hcp-parc_*/*/*.pt',  # In subdirectories
        f'{data_dir}/**/*.pt',  # Recursive
    ]
    
    for pattern in patterns:
        found = glob.glob(pattern, recursive=True)
        pt_files.extend(found)
    
    # Remove duplicates and exclude already-processed files
    pt_files = list(set(pt_files))
    pt_files = [f for f in pt_files if 'hca450_' not in f]  # Exclude processed files
    pt_files.sort()
    
    print(f"Found {len(pt_files)} .pt files")
    
    # Process files and group by subject
    print("\nProcessing files and grouping by subject...")
    subject_data = {}  # {subject_id: {'features': [...], 'label': int}}
    missing_labels = []
    
    for i, filepath in enumerate(pt_files):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(pt_files)} files...")
        
        try:
            subject_id, bold_tensor = process_pt_file(filepath)
            
            # Get label from mapping
            if subject_id in label_mapping:
                label = label_mapping[subject_id]
                
                # Group by subject
                if subject_id not in subject_data:
                    subject_data[subject_id] = {'features': [], 'label': label}
                
                subject_data[subject_id]['features'].append(bold_tensor)
            else:
                missing_labels.append((subject_id, filepath))
        except Exception as e:
            print(f"  Error processing {filepath}: {e}")
            continue
    
    print(f"\nSuccessfully processed {len(pt_files)} files")
    print(f"  Unique subjects: {len(subject_data)}")
    print(f"  Total samples: {sum(len(v['features']) for v in subject_data.values())}")
    
    if missing_labels:
        print(f"\nWarning: {len(missing_labels)} files had missing labels:")
        for sub_id, fpath in missing_labels[:10]:  # Show first 10
            print(f"  Subject {sub_id}: {fpath}")
        if len(missing_labels) > 10:
            print(f"  ... and {len(missing_labels) - 10} more")
    
    if len(subject_data) == 0:
        raise ValueError("No valid files processed!")
    
    # Prepare subject-level data for splitting
    subjects = list(subject_data.keys())
    subject_labels = [subject_data[sub]['label'] for sub in subjects]
    
    print(f"\nSubject-level statistics:")
    print(f"  Total subjects: {len(subjects)}")
    print(f"  Label distribution: {torch.bincount(torch.tensor(subject_labels)).tolist()}")
    
    # Split subjects (not individual samples) into train/val/test
    print(f"\nSplitting subjects (train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%})...")
    
    # First split: train vs (val+test)
    train_subjects, temp_subjects, train_subject_labels, temp_subject_labels = train_test_split(
        subjects, subject_labels,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=subject_labels
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_subjects, test_subjects, val_subject_labels, test_subject_labels = train_test_split(
        temp_subjects, temp_subject_labels,
        test_size=(1 - val_size),
        random_state=seed,
        stratify=temp_subject_labels
    )
    
    # Collect all samples from each subject group
    train_features = []
    train_labels = []
    for sub_id in train_subjects:
        train_features.extend(subject_data[sub_id]['features'])
        train_labels.extend([subject_data[sub_id]['label']] * len(subject_data[sub_id]['features']))
    
    val_features = []
    val_labels = []
    for sub_id in val_subjects:
        val_features.extend(subject_data[sub_id]['features'])
        val_labels.extend([subject_data[sub_id]['label']] * len(subject_data[sub_id]['features']))
    
    test_features = []
    test_labels = []
    for sub_id in test_subjects:
        test_features.extend(subject_data[sub_id]['features'])
        test_labels.extend([subject_data[sub_id]['label']] * len(subject_data[sub_id]['features']))
    
    print(f"\nFinal split statistics:")
    print(f"  Train: {len(train_subjects)} subjects → {len(train_features)} samples")
    print(f"  Val:   {len(val_subjects)} subjects → {len(val_features)} samples")
    print(f"  Test:  {len(test_subjects)} subjects → {len(test_features)} samples")
    print(f"\nSample-level label distribution:")
    print(f"  Train: {torch.bincount(torch.tensor(train_labels)).tolist()}")
    print(f"  Val:   {torch.bincount(torch.tensor(val_labels)).tolist()}")
    print(f"  Test:  {torch.bincount(torch.tensor(test_labels)).tolist()}")
    
    # Save files
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving processed data to {output_dir}/...")
    
    # Save as lists (as expected by the dataset loader)
    # Labels should be a list of integers, not a list of lists
    torch.save(train_features, os.path.join(output_dir, 'hca450_train_x.pt'))
    torch.save(train_labels, os.path.join(output_dir, 'hca450_train_y.pt'))
    
    torch.save(val_features, os.path.join(output_dir, 'hca450_valid_x.pt'))
    torch.save(val_labels, os.path.join(output_dir, 'hca450_valid_y.pt'))
    
    torch.save(test_features, os.path.join(output_dir, 'hca450_test_x.pt'))
    torch.save(test_labels, os.path.join(output_dir, 'hca450_test_y.pt'))
    
    print("✓ Preprocessing complete!")
    print(f"\nFiles saved:")
    print(f"  {output_dir}/hca450_train_x.pt ({len(train_features)} samples)")
    print(f"  {output_dir}/hca450_train_y.pt")
    print(f"  {output_dir}/hca450_valid_x.pt ({len(val_features)} samples)")
    print(f"  {output_dir}/hca450_valid_y.pt")
    print(f"  {output_dir}/hca450_test_x.pt ({len(test_features)} samples)")
    print(f"  {output_dir}/hca450_test_y.pt")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess HCP parcellated data')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing hcp-parc_* subdirectories')
    parser.add_argument('--label_map', type=str, default='hcp_sex_target_id_map.json',
                        help='Path to label mapping JSON file')
    parser.add_argument('--output_dir', type=str, default='brain-jepa-dataset',
                        help='Output directory for processed files')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splitting')
    
    args = parser.parse_args()
    
    preprocess_hcp_data(
        data_dir=args.data_dir,
        label_map_path=args.label_map,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

