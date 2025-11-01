#!/usr/bin/env python3
"""Preprocess HCP parcellated data into expected format for Brain-JEPA"""
import os
import json
import torch
from tqdm import tqdm

try:
    import boto3
    from io import BytesIO
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False
    print("Warning: boto3 not available. S3 upload disabled.")

def load_label_mapping(json_path):
    """Load subject ID to label mapping"""
    with open(json_path, 'r') as f:
        mapping = json.load(f)
    # Keep as string keys (matching meta['sub'] format) and convert values to int
    return {str(k): int(v) for k, v in mapping.items()}

def get_folder_index(folder_name):
    """Extract numeric index from folder name like 'hcp-parc_0000' -> 0"""
    try:
        # Extract number from 'hcp-parc_0000' -> '0000' -> 0
        num_str = folder_name.split('_')[-1]
        return int(num_str)
    except (ValueError, IndexError):
        return None

def list_all_folders(s3_bucket, s3_prefix):
    """List all folders from S3, return folder dict grouped by folder name"""
    if not HAS_BOTO3:
        raise ImportError("boto3 required for S3 operations")
    
    s3 = boto3.client('s3')
    folders = {}  # {folder_name: [file_info]}
    
    print(f"Listing all folders from s3://{s3_bucket}/{s3_prefix}/...")
    paginator = s3.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.pt') and 'hca450_' not in obj['Key']:
                    # Extract folder name (e.g., 'hcp-parc_0000')
                    parts = obj['Key'].split('/')
                    folder_name = None
                    for part in parts:
                        if part.startswith('hcp-parc_'):
                            folder_name = part
                            break
                    
                    if folder_name:
                        if folder_name not in folders:
                            folders[folder_name] = []
                        folders[folder_name].append({
                            'key': obj['Key'],
                            'size': obj['Size']
                        })
    
    print(f"Found {len(folders)} folders")
    return folders

def split_folders_by_range(folders_dict, train_range, val_range, test_range):
    """Split folders into train/val/test by index ranges"""
    train_folders = []
    val_folders = []
    test_folders = []
    
    for folder_name in folders_dict.keys():
        folder_idx = get_folder_index(folder_name)
        if folder_idx is None:
            continue
        
        if train_range[0] <= folder_idx < train_range[1]:
            train_folders.append(folder_name)
        elif val_range[0] <= folder_idx < val_range[1]:
            val_folders.append(folder_name)
        elif test_range[0] <= folder_idx < test_range[1]:
            test_folders.append(folder_name)
    
    return train_folders, val_folders, test_folders

def process_and_filter_files(s3, s3_bucket, folder_list, folders_dict, label_mapping, split_name):
    """Process files from folders, apply strict filtering"""
    subject_data = {}
    stats = {
        'total_files': 0,
        'removed_7t': 0,
        'removed_too_short': 0,
        'truncated_too_long': 0,
        'removed_no_label': 0,
        'kept': 0
    }
    
    # Collect all files from folders in this split
    all_files = []
    for folder_name in folder_list:
        all_files.extend(folders_dict[folder_name])
    
    stats['total_files'] = len(all_files)
    
    for file_info in tqdm(all_files, desc=f"Processing {split_name}"):
        s3_key = file_info['key']
        
        try:
            # Load file from S3
            response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
            data = torch.load(BytesIO(response['Body'].read()), map_location='cpu')
            
            # Check if 3T (remove 7T)
            if data.get('meta', {}).get('mag') != '3T':
                stats['removed_7t'] += 1
                continue
            
            # Extract subject ID
            subject_id = str(data['meta']['sub'])
            
            # Extract bold signal [n_frames, 450]
            bold = data['bold']  # Shape: [n_frames, 450]
            n_frames = bold.shape[0]
            
            # Filter: remove runs < 490 frames
            if n_frames < 490:
                stats['removed_too_short'] += 1
                continue
            
            # Transpose to [450, n_frames] to match expected format (ROIs x timepoints)
            bold = bold.T  # Now shape: [450, n_frames]
            
            # Truncate: take only first 490 frames if > 490
            if n_frames > 490:
                bold = bold[:, :490]  # Take first 490 frames
                stats['truncated_too_long'] += 1
            else:
                # Exactly 490 frames, no truncation needed
                pass
            
            # Check if subject has label
            if subject_id not in label_mapping:
                stats['removed_no_label'] += 1
                continue
            
            label = label_mapping[subject_id]
            
            # Group by subject
            if subject_id not in subject_data:
                subject_data[subject_id] = {'features': [], 'label': label}
            
            subject_data[subject_id]['features'].append(bold.float())
            stats['kept'] += 1
            
        except Exception as e:
            print(f"\n  Error processing {s3_key}: {e}")
            continue
    
    return subject_data, stats

def preprocess_hcp_data(
    s3_bucket='medarc',
    s3_prefix='fmri-fm/datasets/hcp-parc-v2',
    s3_output_prefix=None,  # If None, use same as input
    label_map_path='hcp_sex_target_id_map.json',
    output_dir=None,  # Local output dir (optional, for debugging)
    train_range=(0, 1800),  # Folder indices for train split
    val_range=(1800, 1900),  # Folder indices for val split
    test_range=(1900, 2000),  # Folder indices for test split
):
    """Preprocess HCP data directly from S3, write results back to S3"""
    if not HAS_BOTO3:
        raise ImportError("boto3 required for S3 operations")
    
    s3 = boto3.client('s3')
    s3_output_prefix = s3_output_prefix or s3_prefix
    
    print(f"\n{'='*60}")
    print("PREPROCESSING FROM S3")
    print(f"{'='*60}")
    print(f"Input:  s3://{s3_bucket}/{s3_prefix}/")
    print(f"Output: s3://{s3_bucket}/{s3_output_prefix}/")
    print(f"Train folders: {train_range[0]}:{train_range[1]}")
    print(f"Val folders:   {val_range[0]}:{val_range[1]}")
    print(f"Test folders:  {test_range[0]}:{test_range[1]}")
    print(f"{'='*60}\n")
    
    # Step 1: List all folders from S3
    print("Step 1: Listing all folders from S3...")
    folders_dict = list_all_folders(s3_bucket, s3_prefix)
    
    if len(folders_dict) == 0:
        raise ValueError("No folders found in S3!")
    
    # Step 2: Split folders by index ranges
    print("\nStep 2: Splitting folders by index ranges...")
    train_folders, val_folders, test_folders = split_folders_by_range(
        folders_dict, train_range, val_range, test_range
    )
    
    # Count files before filtering
    train_files_before = sum(len(folders_dict[f]) for f in train_folders)
    val_files_before = sum(len(folders_dict[f]) for f in val_folders)
    test_files_before = sum(len(folders_dict[f]) for f in test_folders)
    
    print(f"\nFolder split:")
    print(f"  Train folders: {len(train_folders)} ({train_range[0]}:{train_range[1]}) → {train_files_before} files")
    print(f"  Val folders:   {len(val_folders)} ({val_range[0]}:{val_range[1]}) → {val_files_before} files")
    print(f"  Test folders:  {len(test_folders)} ({test_range[0]}:{test_range[1]}) → {test_files_before} files")
    
    # Load label mapping
    print(f"\nLoading label mapping from {label_map_path}...")
    label_mapping = load_label_mapping(label_map_path)
    print(f"Loaded {len(label_mapping)} subject labels")
    
    # Step 3: Process and filter files (remove 7T, < 490 frames; truncate > 490 frames)
    print("\nStep 3: Processing and filtering files...")
    print("  - Remove 7T, keep only 3T")
    print("  - Remove runs < 490 frames")
    print("  - Truncate runs > 490 frames to first 490 frames")
    print("  - Match to sex labels")
    
    train_subject_data, train_stats = process_and_filter_files(
        s3, s3_bucket, train_folders, folders_dict, label_mapping, "train"
    )
    val_subject_data, val_stats = process_and_filter_files(
        s3, s3_bucket, val_folders, folders_dict, label_mapping, "val"
    )
    test_subject_data, test_stats = process_and_filter_files(
        s3, s3_bucket, test_folders, folders_dict, label_mapping, "test"
    )
    
    # Print filtering statistics
    def print_stats(stats, split_name):
        print(f"\n{split_name.upper()} filtering stats:")
        print(f"  Total files:          {stats['total_files']}")
        print(f"  Removed (7T):         {stats['removed_7t']}")
        print(f"  Removed (< 490):      {stats['removed_too_short']}")
        print(f"  Truncated (> 490):    {stats['truncated_too_long']}")
        print(f"  Removed (no label):   {stats['removed_no_label']}")
        print(f"  Kept:                 {stats['kept']}")
    
    print_stats(train_stats, "train")
    print_stats(val_stats, "val")
    print_stats(test_stats, "test")
    
    # Step 4: Aggregate samples from subjects
    print("\nStep 4: Aggregating samples...")
    train_features = []
    train_labels = []
    for sub_id, sub_data in train_subject_data.items():
        train_features.extend(sub_data['features'])
        train_labels.extend([sub_data['label']] * len(sub_data['features']))
    
    val_features = []
    val_labels = []
    for sub_id, sub_data in val_subject_data.items():
        val_features.extend(sub_data['features'])
        val_labels.extend([sub_data['label']] * len(sub_data['features']))
    
    test_features = []
    test_labels = []
    for sub_id, sub_data in test_subject_data.items():
        test_features.extend(sub_data['features'])
        test_labels.extend([sub_data['label']] * len(sub_data['features']))
    
    print(f"\nFinal split statistics:")
    print(f"  Train: {len(train_subject_data)} subjects → {len(train_features)} samples")
    print(f"  Val:   {len(val_subject_data)} subjects → {len(val_features)} samples")
    print(f"  Test:  {len(test_subject_data)} subjects → {len(test_features)} samples")
    print(f"\nSample-level label distribution:")
    print(f"  Train: {torch.bincount(torch.tensor(train_labels)).tolist()}")
    print(f"  Val:   {torch.bincount(torch.tensor(val_labels)).tolist()}")
    print(f"  Test:  {torch.bincount(torch.tensor(test_labels)).tolist()}")
    
    # Step 5: Save files
    print("\nStep 5: Saving processed files...")
    
    # Save locally first (for debugging/verification)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving locally to {output_dir}/...")
        torch.save(train_features, os.path.join(output_dir, 'hca450_train_x.pt'))
        torch.save(train_labels, os.path.join(output_dir, 'hca450_train_y.pt'))
        torch.save(val_features, os.path.join(output_dir, 'hca450_valid_x.pt'))
        torch.save(val_labels, os.path.join(output_dir, 'hca450_valid_y.pt'))
        torch.save(test_features, os.path.join(output_dir, 'hca450_test_x.pt'))
        torch.save(test_labels, os.path.join(output_dir, 'hca450_test_y.pt'))
    
    # Upload to S3
    print(f"Uploading to s3://{s3_bucket}/{s3_output_prefix}/...")
    files_to_upload = [
        ('hca450_train_x.pt', train_features),
        ('hca450_train_y.pt', train_labels),
        ('hca450_valid_x.pt', val_features),
        ('hca450_valid_y.pt', val_labels),
        ('hca450_test_x.pt', test_features),
        ('hca450_test_y.pt', test_labels),
    ]
    
    uploaded = 0
    for filename, data in tqdm(files_to_upload, desc="Uploading", unit="file"):
        s3_key = f"{s3_output_prefix}/{filename}"
        
        # Save to BytesIO buffer
        buffer = BytesIO()
        torch.save(data, buffer)
        buffer.seek(0)
        
        try:
            s3.upload_fileobj(buffer, s3_bucket, s3_key)
            uploaded += 1
            print(f"  ✓ Uploaded {filename} → s3://{s3_bucket}/{s3_key}")
        except Exception as e:
            print(f"  ✗ Error uploading {filename}: {e}")
    
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Uploaded: {uploaded}/{len(files_to_upload)} files")
    print(f"Location: s3://{s3_bucket}/{s3_output_prefix}/")
    if output_dir:
        print(f"Local copy: {output_dir}/")
    print(f"{'='*60}")
    
    return {
        'train_features': train_features,
        'train_labels': train_labels,
        'val_features': val_features,
        'val_labels': val_labels,
        'test_features': test_features,
        'test_labels': test_labels,
        's3_bucket': s3_bucket,
        's3_prefix': s3_output_prefix
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess HCP parcellated data from S3')
    
    # S3 arguments
    parser.add_argument('--s3_bucket', type=str, default='medarc',
                        help='S3 bucket name')
    parser.add_argument('--s3_prefix', type=str, default='fmri-fm/datasets/hcp-parc-v2',
                        help='S3 prefix/path')
    parser.add_argument('--s3_output_prefix', type=str, default=None,
                        help='S3 output prefix aka where in s3 we write to (default: same as input prefix)')
    
    # Folder-based splitting arguments
    def parse_range(arg):
        try:
            start, end = map(int, arg.split(':'))
            return (start, end)
        except:
            raise argparse.ArgumentTypeError("Range must be in format start:end (e.g., 0:1800)")
    
    parser.add_argument('--train_range', type=parse_range, default=(0, 100),
                        help='Train folder range (default: 0:1800)')
    parser.add_argument('--val_range', type=parse_range, default=(1800, 1850),
                        help='Val folder range (default: 1800:1900)')
    parser.add_argument('--test_range', type=parse_range, default=(1900, 1950),
                        help='Test folder range (default: 1900:2000)')
    
    # Common arguments
    parser.add_argument('--label_map', type=str, default='hcp_sex_target_id_map.json',
                        help='Path to label mapping JSON file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Local output directory (optional, for debugging)')
    
    args = parser.parse_args()
    
    # Process from S3 → write to S3
    preprocess_hcp_data(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        s3_output_prefix=args.s3_output_prefix,
        label_map_path=args.label_map,
        output_dir=args.output_dir,
        train_range=args.train_range,
        val_range=args.val_range,
        test_range=args.test_range
    )
