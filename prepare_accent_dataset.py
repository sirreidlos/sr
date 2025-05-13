"""
Data Preparation for Accent Classification Dataset

This script helps prepare a dataset for accent classification by:
1. Processing audio files from common accent datasets
2. Creating a CSV file with file paths and accent labels
3. Splitting data into train, validation, and test sets
"""

import os
import re
import pandas as pd
import torch
import torchaudio
import argparse
import random
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def process_audio_file(audio_file, accent_dataset_dir, output_dir, target_sr=16000):
    """
    Process a single audio file.
    
    Args:
        audio_file: Name of the audio file
        accent_dataset_dir: Path to accent dataset
        output_dir: Where to save processed files
        target_sr: Target sample rate
    
    Returns:
        Dict with file information or None if error
    """
    accent = re.match(r"([a-zA-Z]+)\d+\.(?:mp3|wav|flac)", audio_file)
    if accent is None:
        return None
    
    accent = accent.group(1)
    file_name = os.path.splitext(os.path.basename(audio_file))
    
    try:
        audio_path = os.path.join(accent_dataset_dir, audio_file)
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        # Clip Silence
        waveform = torchaudio.functional.vad(waveform, sample_rate)
        
        # Save processed audio
        output_filename = f"{file_name[0]}.wav"
        output_path = os.path.join(output_dir, output_filename)
        torchaudio.save(output_path, waveform, target_sr)
        
        # Return data
        return {
            "file_path": output_path,
            "accent": accent,
            "duration": waveform.shape[1] / target_sr,
            "source": "dataset"
        }
        
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None


def process_accent_dataset(accent_dataset_dir, output_dir, target_sr=16000, num_workers=8):
    """
    Process a generic accent dataset with the following structure:
    accent_dataset_dir/
        accent_a1.mp3
        accent_a2.mp3
        accent_b1.mp3
            ...
    
    Args:
        accent_dataset_dir: Path to accent dataset
        output_dir: Where to save processed files
        target_sr: Target sample rate
        num_workers: Number of parallel workers (defaults to CPU count)
    
    Returns:
        DataFrame with file paths and accent information
    """
    print(f"Processing accent dataset from {accent_dataset_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of audio files
    audio_files = [f for f in os.listdir(accent_dataset_dir) 
                  if re.match(r"[a-zA-Z]+\d+\.(?:mp3|wav|flac)", f)]
    
    # Set up multiprocessing
    num_workers = max(min(mp.cpu_count(), num_workers), 1)
    # num_workers = 1
    
    # Create a partial function with fixed arguments
    process_func = partial(
        process_audio_file,
        accent_dataset_dir=accent_dataset_dir,
        output_dir=output_dir,
        target_sr=target_sr
    )
    
    # Process files in parallel
    print(f"Processing {len(audio_files)} audio files using {num_workers} workers")
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, audio_files),
            total=len(audio_files),
            desc="Processing files",
            smoothing=0.05
        ))
    
    # Filter out None results and create DataFrame
    data = [result for result in results if result is not None]
    return pd.DataFrame(data)


def pitch_shift_resample(waveform, sample_rate, n_steps):
    """
    Approximate pitch shifting using resampling.
    """
    factor = 2 ** (n_steps / 12)
    new_sample_rate = int(sample_rate * factor)
    resample_up = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    resample_down = torchaudio.transforms.Resample(orig_freq=new_sample_rate, new_freq=sample_rate)
    return resample_down(resample_up(waveform))

def time_stretch_spectrogram(waveform, sample_rate, rate):
    """
    Time stretch using STFT and Griffin-Lim.
    """
    n_fft = 1024
    hop_length = 256
    spec_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)
    spec = spec_transform(waveform)

    time_stretch = torchaudio.transforms.TimeStretch(hop_length=hop_length, n_freq=spec.shape[1])
    stretched_spec = time_stretch(spec, rate)

    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=hop_length)
    return griffin_lim(stretched_spec.abs())


def generate_augmented_audio(waveform, sample_rate, max_duration=None):
    """
    Apply various audio augmentations with safe time stretch bounds.

    Args:
        waveform: Input waveform tensor
        sample_rate: Sampling rate
        num_augmentations: Number of augmentations
        max_duration: Max allowed duration in seconds (optional)

    Returns:
        List of augmented waveforms
    """
    original_length = waveform.shape[1]
    max_samples = int(max_duration * sample_rate) if max_duration else None

    aug_type = random.choice(['time_stretch', 'pitch_shift', 'noise', 'time_mask', 'freq_mask'])

    if aug_type == 'time_stretch':
        # Calculate min stretch factor to stay within max duration
        if max_samples:
            min_stretch = max(original_length / max_samples, 0.8)
            max_stretch = 1.2
            if min_stretch > max_stretch:
                min_stretch = max_stretch  # Clamp
            stretch_factor = random.uniform(min_stretch, max_stretch)
        else:
            stretch_factor = random.uniform(0.8, 1.2)

        aug_waveform = time_stretch_spectrogram(waveform, sample_rate, rate=stretch_factor)

    elif aug_type == 'pitch_shift':
        semitone_shift = random.uniform(-2, 2)
        aug_waveform = pitch_shift_resample(waveform, sample_rate, semitone_shift)

    elif aug_type == 'noise':
        noise_level = random.uniform(0.001, 0.005)
        noise = torch.randn_like(waveform) * noise_level
        aug_waveform = waveform + noise

    elif aug_type == 'time_mask':
        mask_length = random.randint(100, 200)
        mask_start = random.randint(0, max(0, waveform.shape[1] - mask_length))
        aug_waveform = waveform.clone()
        aug_waveform[:, mask_start:mask_start + mask_length] = 0

    else:  # freq_mask
        n_fft = 1024
        spec = torchaudio.transforms.Spectrogram(n_fft=n_fft)(waveform)
        mask_length = random.randint(10, 20)
        mask_start = random.randint(0, max(0, spec.shape[2] - mask_length))
        spec[:, :, mask_start:mask_start + mask_length] = 0
        aug_waveform = torchaudio.transforms.GriffinLim(n_fft=n_fft)(spec)

    return aug_waveform


def cleanup_unused_audio_files(processed_dir, accent_data):
    """
    Delete audio files in the processed directory that are not referenced in accent_data DataFrame.
    
    Args:
        processed_dir: Directory containing processed audio files
        accent_data: DataFrame with file_path column containing paths to files that should be kept
    
    Returns:
        int: Number of files deleted
    """
    
    print("Cleaning up unused audio files...")
    
    # Get list of all audio files in the processed directory (and subdirectories)
    all_audio_files = set()
    dataset_dir = os.path.join(processed_dir, "dataset")
    
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):
                all_audio_files.add(os.path.abspath(os.path.join(root, file)))
    
    # Get set of audio files referenced in accent_data
    used_audio_files = set(os.path.abspath(path) for path in accent_data["file_path"])
    
    # Find files to delete (files in all_audio_files but not in used_audio_files)
    files_to_delete = all_audio_files - used_audio_files
    
    # Delete unused files
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    print(f"Deleted {len(files_to_delete)} unused audio files")
    return len(files_to_delete)

def create_dataset(args):
    """Create and prepare the dataset"""
    set_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    processed_dir = os.path.join(args.output_dir, "processed_audio")
    os.makedirs(processed_dir, exist_ok=True)
    
    if not args.accent_dataset_dir:
        print("No input dir!")
        return
    
    accent_data = process_accent_dataset(
        args.accent_dataset_dir, 
        os.path.join(processed_dir, "dataset"),
        args.sample_rate,
        args.num_workers
    )

    # Filter by duration
    if args.min_duration > 0:
        accent_data = accent_data[accent_data["duration"] >= args.min_duration]
    
    if args.max_duration > 0:
        accent_data = accent_data[accent_data["duration"] <= args.max_duration]
    
    # Filter by accent count
    accent_counts = accent_data["accent"].value_counts()
    valid_accents = accent_counts[accent_counts >= args.min_samples_per_accent].index
    
    # print(f"Accent counts before filtering: {accent_counts}")
    for accent, count in accent_counts.items():
        print(f"{accent} : {count}")
    print(f"Valid accents (with at least {args.min_samples_per_accent} samples): {valid_accents.tolist()}")
    
    accent_data = accent_data[accent_data["accent"].isin(valid_accents)]
    accent_data = accent_data.sort_values(by='file_path').reset_index(drop=True)

    if args.augment_dataset_num != 0:
        print("Applying data augmentation...")
        augmented_data = []
        
        # Get accent counts after filtering
        accent_counts = accent_data["accent"].value_counts()
        max_count = accent_counts.max()
        
        for accent in tqdm(valid_accents, desc="Accents"):
            accent_samples = accent_data[accent_data["accent"] == accent]
            current_count = len(accent_samples)
            
            if current_count >= max_count:
                continue

            augment_times = min(args.augment_dataset_num, max_count) - current_count
            
            for _ in tqdm(range(augment_times), desc=f"Augmenting {accent}", leave=False):
                sample = accent_samples.sample(1, random_state=int(random.randint(0, int(1e9)))).squeeze()
                waveform, sample_rate = torchaudio.load(sample["file_path"])
                aug_waveform = generate_augmented_audio(waveform, sample_rate, max_duration=args.max_duration)
                curr_accent_count = accent_data[accent_data["accent"] == accent].size
                aug_filename = f"{accent}{int(int(curr_accent_count )+ int(len(augmented_data)) + int(1))}.wav"
                aug_path = os.path.join(processed_dir, "dataset", aug_filename)
                torchaudio.save(aug_path, aug_waveform, sample_rate)
                
                # Add to augmented data
                augmented_data.append({
                    "file_path": aug_path,
                    "accent": accent,
                    "duration": aug_waveform.shape[1] / sample_rate,
                    "source": os.path.basename(sample["file_path"])
                })
                

        # Add augmented data to the main dataset
        if augmented_data:
            accent_data = pd.concat([accent_data, pd.DataFrame(augmented_data)], ignore_index=True)
            print(f"Added {len(augmented_data)} augmented samples")
    
    if args.oversample != 0:
        accent_counts = accent_data["accent"].value_counts()

        for accent in tqdm(valid_accents, desc="Oversampling Accents"):
            accent_samples = accent_data[accent_data["accent"] == accent]
            current_count = len(accent_samples)

            if current_count >= args.oversample:
                continue

            needed = args.oversample - current_count
            extra_samples = accent_samples.sample(n=needed, replace=True, random_state=args.seed)
            accent_data = pd.concat([accent_data, extra_samples], ignore_index=True)

    # Balance dataset if requested
    accent_counts = accent_data["accent"].value_counts()
    if args.balance_dataset:
        min_count = accent_counts[valid_accents].min()
        print(f"MIN: {min_count}")
        balanced_data = []
        
        for accent in valid_accents:
            curr_accent_data = accent_data[accent_data["accent"] == accent]
            print(f"PROCESSING {accent} WITH {len(curr_accent_data)} CLIPS")
            if len(curr_accent_data) > min_count:
                print(f"DOWNSAMPLED {accent}")
                curr_accent_data = curr_accent_data.sample(min_count, random_state=args.seed)
            balanced_data.append(curr_accent_data)
            
        print("BALANCED THE DATA SOMEWHAT")
        print(balanced_data)
        accent_data = pd.concat(balanced_data, ignore_index=True)
    
    cleanup_unused_audio_files(processed_dir, accent_data)
    
    # Split into train, validation, test
    accents = accent_data["accent"].unique()
    
    # Stratify by accent
    train_data, temp_data = train_test_split(
        accent_data, 
        test_size=args.val_size + args.test_size,
        stratify=accent_data["accent"],
        random_state=args.seed
    )
    
    # Adjust validation and test sizes
    val_ratio = args.val_size / (args.val_size + args.test_size)
    val_data, test_data = train_test_split(
        temp_data, 
        test_size=1-val_ratio,
        stratify=temp_data["accent"],
        random_state=args.seed
    )
    
    # Add split column
    train_data["split"] = "train"
    val_data["split"] = "val"
    test_data["split"] = "test"
    
    # Combine and save
    final_data = pd.concat([train_data, val_data, test_data], ignore_index=True)
    
    # Map accent names to standardized names if provided
    if args.accent_mapping:
        try:
            accent_map = {}
            with open(args.accent_mapping, 'r') as f:
                for line in f:
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        accent_map[parts[0].strip()] = parts[1].strip()
            
            final_data["original_accent"] = final_data["accent"]
            final_data["accent"] = final_data["accent"].map(accent_map).fillna(final_data["accent"])
            
        except Exception as e:
            print(f"Error applying accent mapping: {e}")
    
    # Save to CSV
    output_path = os.path.join(args.output_dir, "accent_data.csv")
    final_data.to_csv(output_path, index=False)
    
    print(f"Dataset prepared and saved to {output_path}")
    print(f"Total samples: {len(final_data)}")
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Accent distribution:\n{final_data['accent'].value_counts()}")
    print("RAND: ", int(random.randint(0, int(1e9))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for accent classification")
    
    parser.add_argument("--output_dir", type=str, default="./accent_dataset",
                        help="Directory to save the processed dataset")
    parser.add_argument("--accent_dataset_dir", type=str, default="",
                        help="Path to custom accent dataset")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="Target sample rate for audio files")
    parser.add_argument("--min_duration", type=float, default=1.0,
                        help="Minimum duration of audio files in seconds")
    parser.add_argument("--max_duration", type=float, default=30.0,
                        help="Maximum duration of audio files in seconds")
    parser.add_argument("--min_samples_per_accent", type=int, default=50,
                        help="Minimum number of samples required per accent")
    parser.add_argument("--balance_dataset", action="store_true",
                        help="Balance dataset by undersampling majority classes")
    parser.add_argument("--augment_dataset_num", type=int, default=0,
                        help="Number of new data to create by performing dataset augmentation for lower count accents")
    parser.add_argument("--oversample", type=int, default=0,
                        help="Number of oversampling done for lower count accents")
    parser.add_argument("--train_size", type=float, default=0.7,
                        help="Proportion of data to use for training")
    parser.add_argument("--val_size", type=float, default=0.15,
                        help="Proportion of data to use for validation")
    parser.add_argument("--test_size", type=float, default=0.15,
                        help="Proportion of data to use for testing")
    parser.add_argument("--accent_mapping", type=str, default="",
                        help="Path to file mapping original accent names to standardized names")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of worker threads")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.accent_dataset_dir]):
        parser.error("At least one dataset source must be provided.")
    
    create_dataset(args)
