import csv
import numpy as np
# import os
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

import pandas as pd
import numpy as np

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
