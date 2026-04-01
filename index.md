# Day logs

## 2026-03-18
Set up Github account.
Ordered [these diodes](https://www.amazon.com/dp/B0CZRH2DHJ?ref=ppx_yo2ov_dt_b_fed_asin_title) given the backorder status of the [Sparkfun diodes](https://www.sparkfun.com/sparkfun-ambient-light-sensor-breakout-temt6000.html) ones Dr. Bergés recommended.

National Instruments documentation confirms that the proper driver for both the NI-9234/9162 unit and the NI  myDAQ is the DAQmx driver, which can be used without needing LabView. There's a Python module called [PyDAQmx](https://pypi.org/project/PyDAQmx) which we may try out first.

After some wiring adventures got the photoresistor to convert to digital data with NI myDAQ. Integrated into Python to record and manipulate data with PyDAQmx. Data appears reasonable.

## 2026-03-25

Learned that we had mistakenly been using the ('floating') digital ground port instead of the analog one on the MyDAQ, so we swapped to the analog ground, and also tried out one of the photodiodes, which did not immediately behave as expected, despite purporting to take the same wiring configuration. Unless we had our wires accidentally crossed, we suspect that the (lowercase-D) dupont connectors aren't engaging well for the photodiodes, which may mean soldering at least one set of ends. TechSpark is woefully lacking in certain basic equipment at this point, and didn't even have precision screwdrivers this evening, whereas they did. I (PEKO) likely need to bring in my own drivers.

Also discovered that the MyDAQ can sample from analog signals at up to 200,000 Hz, though no higher. Interestingly, the NI-9234/9162's documentation indicates a maxiumum sampling rate of 51,200 samples per second, for comparison, though the samples themselves should be of higher quality (higher instantaneous discrimination of levels -- up to 24, versus the MyDAQ's 16 -- less noisy, etc.) and presumably, this rate is independent of the number of input sources connected (up to 4).

## 2026-03-28

Found a compact fluorescent bulb and two types of LED bulbs to test with.

## 2026-03-29

Bought two different kinds of 40-W incandescent bulbs for baseline test. Waiting on lamp!

## 2026-03-31

Obtained lamp and headed to TechSpark to attempt soldering one of the photodiodes. (PEKO) completed the obligatory soldering safety training and soldered one photodiode assembly.

### 2026-03-31 Notes

Used the breadboard and one of the other trio pin bundles to help stabilize and level the piece. (The other set of pins can just be stuck into the breadboard in the normal orientation such that the plastic collar holds up the other end of the little board.) Irons in TechSpark can be set to a specifc temperature. 425 &deg; seemed to be the minimum temperature needed when heating only a pin, per the approach that appears to be recommended to avoid a 'cold solder joint,' which may work poorly. AK to collect initial data tomorrow so we can have a more meaningful discussion with Dr. B on Thursday.
