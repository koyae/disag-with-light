import csv
import json

import torch
import torch.nn as nn
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


class NILMDataset(Dataset):
    def __init__(self, voltage_df, events_df, target_devices, window_size=500):
        self.window_size = window_size
        self.voltage = voltage_df['voltage_V'].values

        # 1. Create a continuous state matrix (0s and 1s) for every voltage sample
        self.state_matrix = np.zeros((len(voltage_df), len(target_devices)), dtype=np.float32)

        # 2. Replay the events to fill the state matrix
        current_state = {dev: 0 for dev in target_devices}

        # Assuming events_df is sorted by elapsed_s
        event_idx = 0
        num_events = len(events_df)

        for i, elapsed_s in enumerate(voltage_df['elapsed_s'].values):
            # Check if we've passed the timestamp of the next event
            while event_idx < num_events and elapsed_s >= events_df.iloc[event_idx]['elapsed_s']:
                event_str = events_df.iloc[event_idx]['label']

                # Parse "device_on" or "device_off"
                for dev_idx, dev in enumerate(target_devices):
                    if dev in event_str:
                        if "_on" in event_str:
                            current_state[dev] = 1.0
                        elif "_off" in event_str:
                            current_state[dev] = 0.0
                event_idx += 1

            # Record the current state for this sample
            for dev_idx, dev in enumerate(target_devices):
                self.state_matrix[i, dev_idx] = current_state[dev]

    def __len__(self):
        # Number of sliding windows we can extract
        return len(self.voltage) - self.window_size

    def __getitem__(self, idx):
        # Extract a window of voltage data
        x = self.voltage[idx : idx + self.window_size]

        # Get the labels for this window.
        # (Usually, we take the state at the end of the window, or the mode)
        y = self.state_matrix[idx + self.window_size - 1]

        # Convert to PyTorch tensors
        # LSTM expects shape: (Sequence Length, Number of Features)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return x_tensor, y_tensor


class BaselineLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=4):
        super(BaselineLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # The LSTM layer
        # batch_first=True means inputs should be (batch_size, sequence_length, features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # A fully connected layer to map the LSTM hidden state to our device classes
        self.fc = nn.Linear(hidden_size, num_classes)

        # Sigmoid for multi-label classification (outputs probabilities between 0 and 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the LAST time step in the window
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)

        return out

