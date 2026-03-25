import time
import nidaqmx
from nidaqmx.constants import TerminalConfiguration

CHANNEL = "myDAQ1/ai0"

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan(
        CHANNEL,
        terminal_config=TerminalConfiguration.DIFF,
        min_val=0.0,
        max_val=5.0,
    )

    while True:
        value = task.read()
        print(f"{value:.3f} V")
        time.sleep(0.1)