import csv
import json

import torch

from torch.utils.data import Dataset

import numpy as np
import os
import pandas as pd
import pprint


def get_sample_rate(file_name):
    with open(file_name,'r') as rh:
        # advance 1000 lines into the CSV, then read 10:
        cr = csv.reader(rh)
        # skip header
        next(cr) # [sample_index, elapsed_s, voltage_V]
        sample = []
        t0 = float(next(cr)[1])
        t1 = float(next(cr)[1])
        return round(1 / (t1 - t0))


def downsample_with_pandas(input_file, factor, output_file=None, chunk_size=None, chunk_duration=None, skip_seconds=0, n_seconds=None) -> None|pd.DataFrame:
    """
    Downsamples a large CSV by averaging every `factor` rows.
    Writes to file or just returns the sampled file/section as a DataFrame.
    """
    caller_args = locals().copy()
    sample_rate = get_sample_rate(input_file)

    if chunk_size and chunk_duration:
        raise ValueError("Specify either chunk_size or chunk_duration, not both.")

    old_chunk_size = chunk_size
    if not chunk_size and not chunk_duration:
        chunk_size = 100_000
    elif chunk_duration and not chunk_size:
        chunk_size = int(chunk_duration * sample_rate)
        print(f"Chunk size from chunk_duration @ {sample_rate} Hz is {chunk_size}")

    chunk_size = (chunk_size // factor) * factor
    if old_chunk_size and old_chunk_size != chunk_size:
        print(
            f"Rounded chunk size from {old_chunk_size} to {chunk_size} to match downsampling ({factor})"
        )

    # Calculate exact rows to skip
    rows_to_skip = int(skip_seconds * sample_rate)

    # Pass a range to skiprows so it skips data (rows 1 to N) but keeps row 0 (header)
    skiprows = range(1, rows_to_skip + 1) if rows_to_skip is not None and rows_to_skip > 0 else None

    # Cast to integer, as read_csv requires whole numbers
    nrows = int(n_seconds * sample_rate) if n_seconds is not None else None

    first_chunk = True
    accumulated_chunks = []

    # pd.read_csv with chunksize returns an iterator, keeping memory usage tiny
    for chunk in pd.read_csv(input_file, chunksize=chunk_size, skiprows=skiprows, nrows=nrows):

        # Group every 'factor' rows together and take the mean
        # e.g., rows 0-9 become one row, 10-19 become the next row
        group_ids = np.arange(len(chunk)) // factor
        downsampled_chunk = chunk.groupby(group_ids).mean()

        # # If sample_index needs to stay an integer, convert it back:
        # if 'sample_index' in downsampled_chunk.columns:
        #     downsampled_chunk['sample_index'] = downsampled_chunk['sample_index'].astype(int)

        if output_file:
            downsampled_chunk.to_csv(
                output_file,
                mode='w' if first_chunk else 'a',
                header=first_chunk,
                index=False
            )
        else:
            accumulated_chunks.append(downsampled_chunk)

        first_chunk = False
        print(f"Processed chunk...")

    if not output_file:
        if accumulated_chunks:
            # pd.concat merges our list of tiny dataframes into one big one
            final_df = pd.concat(accumulated_chunks, ignore_index=True)
            if not final_df.empty:
                return final_df
            else:
                raise Exception(
                    f"File {input_file} was empty for {pprint.pformat(caller_args)}."
                )

"""
Recommended invocation:

    desc_files = [ f for f in os.listdir("data") if f.startswith("desc_") ]
    event_files_with_descs = [ os.path.splitext("events_"+f[len('desc_'):])[0] + '.csv' for f in desc_files ]
    total_event_count = 0
    for f in event_files_with_descs:
        print(f)
        event_count = count_num_samples(os.path.join('data',f))
        print(f"{f} has {event_count} events.")
        total_event_count += event_count
    print(f"Total events: {total_event_count}")
"""
def count_num_samples(event_csv, harsh=True):
    if harsh:
        basename = os.path.basename(event_csv)
        if not basename.startswith("events_"):
            raise Exception(f"harsh mode: Filename {event_csv} should start with 'events_'")
        if not basename.endswith(".csv"):
            raise Exception(f"harsh mode: Filename {event_csv} should end with '.csv'")
    with open(event_csv, 'r') as rh:
        reader = csv.reader(rh)
        nx = next(reader,None)  # skip header
        if nx is None:
            print(f"Warning: File {event_csv} is empty.")
            return 0
        return sum(1 for row in reader)
import json


def get_target_devices(desc_json_path, exclude_load_types=None):
    if exclude_load_types is None:
        exclude_load_types = []

    with open(desc_json_path, 'r') as f:
        meta = json.load(f)

    target_devices = []
    for device_id, info in meta["devices"].items():
        load_types = info["description"].get("load_types", [])

        # If the device has a load type we want to exclude, skip it
        if any(lt in exclude_load_types for lt in load_types):
            print(f"Excluding {device_id} (Load types: {load_types})")
            continue

        target_devices.append(device_id)

    return target_devices


class FastNILMDataset(Dataset):
    def __init__(self, npz_path, window_size=200):
        # np.load reads compiled binary instantly
        data = np.load(npz_path)
        self.voltage = data['voltage']
        self.state_matrix = data['states']
        self.window_size = window_size

        self.v_mean = np.mean(self.voltage)
        self.v_std = np.std(self.voltage)

        # Watch for 0:
        self.v_std = self.v_std if self.v_std else 1

    def __len__(self):
        return len(self.voltage) - self.window_size

    def __getitem__(self, idx):
        x = self.voltage[idx : idx + self.window_size]
        y = self.state_matrix[idx + self.window_size - 1]

        # slight hack to make sure the uncertain label doesn't mess things up:
        y = np.maximum(y, 0.0)

        x_normalized = (x - self.v_mean) / self.v_std

        x_tensor = torch.tensor(x_normalized, dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return x_tensor, y_tensor



def get_micro_dataset(cache_dir="preprocess_cache"):
    print("[*] Scanning cache for device coverage...")
    npz_files = [f for f in os.listdir(cache_dir) if f.endswith(".npz")]

    file_to_devices = {}
    all_devices_seen = set()

    # 1. Map out which devices are actually ACTIVE in each file
    for f in npz_files:
        filepath = os.path.join(cache_dir, f)
        data = np.load(filepath)

        states = data['states']
        devices = data['devices']

        # Sum down the time axis. If the sum > 0, the device turned ON at least once.
        active_cols = np.where(states.sum(axis=0) > 0)[0]
        active_devices = {devices[i] for i in active_cols}

        if active_devices:
            file_to_devices[f] = active_devices
            all_devices_seen.update(active_devices)

    print(f"[*] Found {len(all_devices_seen)} unique devices across {len(npz_files)} files.")

    # 2. Greedy Set Cover Algorithm
    uncovered_devices = set(all_devices_seen)
    minimum_files = []

    print("\n[*] Assembling minimum file set:")
    while uncovered_devices:
        # Find the file that covers the highest number of currently UNCOVERED devices
        best_file = max(
            file_to_devices.keys(),
            key=lambda f: len(file_to_devices[f].intersection(uncovered_devices))
        )

        # What devices are we adding with this file?
        newly_covered = file_to_devices[best_file].intersection(uncovered_devices)

        minimum_files.append(best_file)
        uncovered_devices -= newly_covered

        print(f" [+] Added {best_file} -> Covered: {list(newly_covered)}")

    print(f"\n[SUCCESS] Reduced {len(npz_files)} files down to {len(minimum_files)} files!")
    return minimum_files

