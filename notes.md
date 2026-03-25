# Day logs

## 2026-03-18
Set up Github account.
Ordered [these diodes](https://www.amazon.com/dp/B0CZRH2DHJ?ref=ppx_yo2ov_dt_b_fed_asin_title) given the backorder status of the [Sparkfun diodes](https://www.sparkfun.com/sparkfun-ambient-light-sensor-breakout-temt6000.html) ones Dr. Bergés recommended.

National Instruments documentation confirms that the proper driver for both the NI-9234/9162 unit and the NI  myDAQ is the DAQmx driver, which can be used without needing LabView. There's a Python module called [PyDAQmx](https://pypi.org/project/PyDAQmx) which we may try out first.

After some wiring adventures got the photoresistor to convert to digital data with NI myDAQ. Integrated into Python to record and manipulate data with PyDAQmx. Data appears reasonable.

