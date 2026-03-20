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
from multiprocessing import Pool, cpu_count

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
) -> Tuple[bool, str, str]:
    """
    Convert Pub1M label to model format and save
    
    Returns:
        Tuple of (success: bool, image_path: str, label_path: str)
        If failed, returns (False, None, output_path)
    """
    try:
        parser = Pub1MParser(
            xml_path=xml_path,
            words_path=words_path,
            image_path=None
        )
        
        data = parser.parse_to_model_format(image_base = image_base_dir)
        
        # Save converted label
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return (True, data['image_path'], str(output_path))
    
    except Exception as e:
        print(f"Error converting {xml_path}: {e}")
        return (False, None, str(output_path))


def _process_single_label(args_tuple):
    """
    Worker function for multiprocessing
    Unpacks arguments and calls convert_and_save_label
    """
    xml_path, words_path, output_path, image_base_dir = args_tuple
    return convert_and_save_label(xml_path, words_path, output_path, image_base_dir)


def create_dataset_split(
    xml_dir: str,
    words_dir: str,
    output_dir: str,
    split_name: str,
    num_samples: int,
    seed: int = 42,
    used_samples: set = None,
    num_workers: int = None,
    image_base_dir: str = None
) -> Tuple[List[Tuple[str, str]], set]:
    """
    Create a dataset split (train/val/test)
    
    Args:
        used_samples: Set of already used sample identifiers to avoid overlap
        num_workers: Number of worker processes (default: cpu_count())
        image_base_dir: Base directory for images (optional)
    
    Returns:
        Tuple of (list of (image_path, label_path) tuples, updated used_samples set)
    """
    print(f"\nCreating {split_name} split ({num_samples} samples)...")
    
    if used_samples is None:
        used_samples = set()
    
    if num_workers is None:
        num_workers = cpu_count()
    
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
    
    # Prepare arguments for parallel processing
    process_args = []
    sample_ids = []
    for xml_path, words_path in sampled_pairs:
        xml_file = Path(xml_path)
        sample_id = xml_file.stem
        label_path = labels_dir / f"{sample_id}.json"
        process_args.append((xml_path, words_path, str(label_path), image_base_dir))
        sample_ids.append(sample_id)
    
    # Process labels in parallel
    print(f"  Processing {num_samples} samples using {num_workers} workers...")
    dataset_list = []
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(_process_single_label, process_args)
    
    # Collect successful results
    for i, (success, image_path, label_path) in enumerate(results):
        if success:
            dataset_list.append((image_path, label_path))
            used_samples.add(sample_ids[i])  # Mark as used
    
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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes for parallel processing (default: number of CPU cores)"
    )
    parser.add_argument(
        "--image_base_dir",
        type=str,
        default=None,
        help="Base directory for images (optional)"
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
    
    num_workers = args.num_workers if args.num_workers else cpu_count() // 2
    print(f"Using {num_workers} worker processes for parallel processing")
    
    if args.num_train > 0:
        train_list, used_samples = create_dataset_split(
            args.xml_dir,
            args.words_dir,
            args.output_dir,
            "train",
            args.num_train,
            seed=args.seed,
            used_samples=used_samples,
            num_workers=num_workers,
            image_base_dir=args.image_base_dir
        )
        save_dataset_list(train_list, output_dir / "train" / "dataset_list.json")

    if args.num_val > 0:
        val_list, used_samples = create_dataset_split(
            args.xml_dir,
            args.words_dir,
            args.output_dir,
            "val",
            args.num_val,
            seed=args.seed,
            used_samples=used_samples,
            num_workers=num_workers,
            image_base_dir=args.image_base_dir
        )
        save_dataset_list(val_list, output_dir / "val" / "dataset_list.json")
    
    if args.num_test > 0:
        test_list, used_samples = create_dataset_split(
            args.xml_dir,
            args.words_dir,
            args.output_dir,
            "test",
            args.num_test,
            seed=args.seed,
            used_samples=used_samples,
            num_workers=num_workers,
            image_base_dir=args.image_base_dir
        )
        save_dataset_list(test_list, output_dir / "test" / "dataset_list.json")
    
    # Save dataset lists
    
    
    
    
    # # Create summary
    # summary = {
    #     "train": len(train_list),
    #     "val": len(val_list),
    #     "test": len(test_list),
    #     "total": len(train_list) + len(val_list) + len(test_list)
    # }
    
    # summary_path = output_dir / "dataset_summary.json"
    # with open(summary_path, 'w') as f:
    #     json.dump(summary, f, indent=2)
    
    # print("\n" + "="*60)
    # print("Dataset Creation Complete!")
    # print("="*60)
    # print(f"Train: {summary['train']} samples")
    # print(f"Val: {summary['val']} samples")
    # print(f"Test: {summary['test']} samples")
    # print(f"Total: {summary['total']} samples")
    # print(f"\nDataset saved to: {args.output_dir}")
    # print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

