import numpy as np
# import os
import pandas as pd

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

def downsample_with_pandas(input_file, output_file=None, factor=10, chunk_size=None, chunk_duration=None, skip_seconds=0, n_seconds=None):
    """
    Downsamples a large CSV by averaging every `factor` rows.
    Writes to file or just returns the sampled file/section.
    """
    # Ensure chunk size is a clean multiple of the downsample factor
    sample_rate = get_sample_rate(input_file)

    if chunk_size and chunk_duration:
        raise ValueError("Specify either chunk_size or chunk_duration, not both.")
    if not chunk_size and not chunk_duration:
        chunk_size = 100_000
    elif chunk_duration and not chunk_size:
        chunk_size = chunk_duration * sample_rate
        print(f"Chunk size from chunk_duration @ {sample_rate} Hz is {chunk_size}")
    chunk_size = (chunk_size // factor) * factor
    print(f"Rounded chunk size to {chunk_size}")

    skiprows = skip_seconds * sample_rate
    nrows = n_seconds * sample_rate if n_seconds is not None else None

    first_chunk = True

    # pd.read_csv with chunksize returns an iterator, keeping memory usage tiny
    for chunk in pd.read_csv(input_file, chunksize=chunk_size, skiprows=skiprows, nrows=nrows):

        # Group every 'factor' rows together and take the mean
        # e.g., rows 0-9 become one row, 10-19 become the next row
        group_ids = np.arange(len(chunk)) // factor
        downsampled_chunk = chunk.groupby(group_ids).mean()

        # If your sample_index needs to stay an integer, convert it back:
        downsampled_chunk['sample_index'] = downsampled_chunk['sample_index'].astype(int)

        if output_file:
            downsampled_chunk.to_csv(
                output_file,
                mode='w' if first_chunk else 'a',
                header=first_chunk,
                index=False
            )
        first_chunk = False
        print(f"Processed chunk...")
