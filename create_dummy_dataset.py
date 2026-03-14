#!/usr/bin/env python3
"""
Create dummy dataset from PubTables1M
Samples data and converts labels to model format
"""
import argparse
import sys
from pathlib import Path
import json
import random
import shutil
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tsr.data.pub1m_parser import Pub1MParser


def find_available_samples(xml_dir: str, words_dir: str, max_samples: int = None) -> List[Tuple[str, str]]:
    """
    Find available XML/words pairs from Pub1M dataset
    
    Returns:
        List of (xml_path, words_path) tuples
    """
    xml_dir = Path(xml_dir)
    words_dir = Path(words_dir)
    
    # Find all XML files
    xml_files = list(xml_dir.glob("*.xml"))
    
    available_pairs = []
    
    for xml_file in xml_files:
        # Find corresponding words file
        words_file = words_dir / f"{xml_file.stem}_words.json"
        
        if words_file.exists():
            available_pairs.append((str(xml_file), str(words_file)))
        
        if max_samples and len(available_pairs) >= max_samples:
            break
    
    return available_pairs


def convert_and_save_label(
    xml_path: str,
    words_path: str,
    output_path: str,
    image_base_dir: str = None
) -> bool:
    """
    Convert Pub1M label to model format and save
    
    Returns:
        True if successful, False otherwise
    """
    try:
        parser = Pub1MParser(
            xml_path=xml_path,
            words_path=words_path,
            image_path=None
        )
        
        data = parser.parse_to_model_format()
        
        # Update image path if image_base_dir provided
        if image_base_dir:
            # Try to find image in athena_format directory
            image_filename = Path(data['image_path']).name
            athena_base = Path("/mnt/disks/data/flax/table_data/external/pub1m/org/athena_format/train")
            
            # Search for image
            image_path = None
            if athena_base.exists():
                for subdir in athena_base.iterdir():
                    if subdir.is_dir():
                        candidate = subdir / "input" / image_filename
                        if candidate.exists():
                            image_path = str(candidate)
                            break
            
            if image_path:
                data['image_path'] = image_path
        
        # Save converted label
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True
    
    except Exception as e:
        print(f"Error converting {xml_path}: {e}")
        return False


def create_dataset_split(
    xml_dir: str,
    words_dir: str,
    output_dir: str,
    split_name: str,
    num_samples: int,
    seed: int = 42,
    used_samples: set = None
) -> Tuple[List[Tuple[str, str]], set]:
    """
    Create a dataset split (train/val/test)
    
    Args:
        used_samples: Set of already used sample identifiers to avoid overlap
    
    Returns:
        Tuple of (list of (image_path, label_path) tuples, updated used_samples set)
    """
    print(f"\nCreating {split_name} split ({num_samples} samples)...")
    
    if used_samples is None:
        used_samples = set()
    
    # Find available samples
    available_pairs = find_available_samples(xml_dir, words_dir, max_samples=None)
    
    # Filter out already used samples
    available_pairs = [
        pair for pair in available_pairs
        if Path(pair[0]).stem not in used_samples
    ]
    
    if len(available_pairs) < num_samples:
        print(f"Warning: Only {len(available_pairs)} samples available, requested {num_samples}")
        num_samples = len(available_pairs)
    
    # Sample randomly (use different seed for each split)
    split_seed = seed
    if split_name == "val":
        split_seed = seed + 1000
    elif split_name == "test":
        split_seed = seed + 2000
    
    random.seed(split_seed)
    sampled_pairs = random.sample(available_pairs, num_samples)
    
    output_dir = Path(output_dir)
    labels_dir = output_dir / split_name / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_list = []
    
    for i, (xml_path, words_path) in enumerate(sampled_pairs):
        xml_file = Path(xml_path)
        sample_id = xml_file.stem
        
        # Convert label
        label_path = labels_dir / f"{sample_id}.json"
        
        if convert_and_save_label(xml_path, words_path, str(label_path)):
            # Get image path from converted label
            with open(label_path, 'r') as f:
                label_data = json.load(f)
                image_path = label_data['image_path']
            
            dataset_list.append((image_path, str(label_path)))
            used_samples.add(sample_id)  # Mark as used
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{num_samples} samples...")
    
    print(f"  Completed: {len(dataset_list)}/{num_samples} samples")
    
    return dataset_list, used_samples


def save_dataset_list(dataset_list: List[Tuple[str, str]], output_path: str):
    """Save dataset list (image_path, label_path) pairs"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(dataset_list, f, indent=2)
    
    print(f"Saved dataset list to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create dummy dataset from PubTables1M")
    parser.add_argument(
        "--xml_dir",
        type=str,
        required=True,
        help="Directory containing Pub1M XML files"
    )
    parser.add_argument(
        "--words_dir",
        type=str,
        required=True,
        help="Directory containing Pub1M words JSON files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./dummy_dataset",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=500,
        help="Number of training samples"
    )
    parser.add_argument(
        "--num_val",
        type=int,
        default=100,
        help="Number of validation samples"
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=100,
        help="Number of test samples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Creating Dummy Dataset from PubTables1M")
    print("="*60)
    print(f"XML directory: {args.xml_dir}")
    print(f"Words directory: {args.words_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Train: {args.num_train}, Val: {args.num_val}, Test: {args.num_test}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create splits (ensure no overlap)
    used_samples = set()
    
    train_list, used_samples = create_dataset_split(
        args.xml_dir,
        args.words_dir,
        args.output_dir,
        "train",
        args.num_train,
        seed=args.seed,
        used_samples=used_samples
    )
    
    val_list, used_samples = create_dataset_split(
        args.xml_dir,
        args.words_dir,
        args.output_dir,
        "val",
        args.num_val,
        seed=args.seed,
        used_samples=used_samples
    )
    
    test_list, used_samples = create_dataset_split(
        args.xml_dir,
        args.words_dir,
        args.output_dir,
        "test",
        args.num_test,
        seed=args.seed,
        used_samples=used_samples
    )
    
    # Save dataset lists
    save_dataset_list(train_list, output_dir / "train" / "dataset_list.json")
    save_dataset_list(val_list, output_dir / "val" / "dataset_list.json")
    save_dataset_list(test_list, output_dir / "test" / "dataset_list.json")
    
    # Create summary
    summary = {
        "train": len(train_list),
        "val": len(val_list),
        "test": len(test_list),
        "total": len(train_list) + len(val_list) + len(test_list)
    }
    
    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("Dataset Creation Complete!")
    print("="*60)
    print(f"Train: {summary['train']} samples")
    print(f"Val: {summary['val']} samples")
    print(f"Test: {summary['test']} samples")
    print(f"Total: {summary['total']} samples")
    print(f"\nDataset saved to: {args.output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

