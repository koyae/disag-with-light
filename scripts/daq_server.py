# daq_server.py
import socket
import numpy as np
import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
from nidaqmx.errors import DaqReadError

CHANNEL     = "myDAQ1/ai0"
SAMPLE_RATE = 1_000
BUFFER_SIZE = 1000
HOST        = "0.0.0.0"
PORT        = 9999

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan(
        CHANNEL,
        terminal_config=TerminalConfiguration.DIFF,
        min_val=0.0, max_val=5.0,
    )
    task.timing.cfg_samp_clk_timing(
        rate=SAMPLE_RATE,
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=BUFFER_SIZE * 100,  # increased from 10x to 100x
    )

    print(f"DAQ server listening on {HOST}:{PORT}...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, PORT))
        srv.listen(1)

        while True:
            print("Waiting for connection...")
            conn, addr = srv.accept()
            print(f"Connected: {addr}")
            with conn:
                try:
                    while True:
                        try:
                            samples = task.read(
                                number_of_samples_per_channel=BUFFER_SIZE
                            )
                        except DaqReadError as e:
                            print(f"DAQ read error (buffer overflow) — skipping: {e}")
                            # reset read position by reading all available samples
                            try:
                                task.read(
                                    number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE
                                )
                            except Exception:
                                pass
                            continue

                        data = np.array(samples, dtype=np.float32).tobytes()
                        conn.sendall(len(data).to_bytes(4, "big") + data)

                except (BrokenPipeError, ConnectionResetError):
                    print("Client disconnected — waiting for new connection...")