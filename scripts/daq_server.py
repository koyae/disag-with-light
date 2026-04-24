# daq_server.py (run on Windows)
import socket
import numpy as np
import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType

CHANNEL     = "myDAQ1/ai0"
SAMPLE_RATE = 10_000
BUFFER_SIZE = 1000
HOST        = "0.0.0.0"   # listen on all interfaces
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
        samps_per_chan=BUFFER_SIZE * 10,
    )

    print(f"DAQ server listening on {HOST}:{PORT}...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, PORT))
        srv.listen(1)
        conn, addr = srv.accept()
        print(f"Connected: {addr}")

        with conn:
            try:
                while True:
                    samples = task.read(
                        number_of_samples_per_channel=BUFFER_SIZE
                    )
                    data = np.array(samples, dtype=np.float32).tobytes()
                    # send length prefix then data
                    conn.sendall(len(data).to_bytes(4, "big") + data)
            except (BrokenPipeError, ConnectionResetError):
                print("Client disconnected.")
            except KeyboardInterrupt:
                print("Stopped.")