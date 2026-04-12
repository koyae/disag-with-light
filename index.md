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

## 2026-04-01

Collected data from variety of appliances (kettle, toaster, heat gun, fridge, microwave) and bulb types (incandescent, CFL, LED). Load signatures typically visible.

## 2026-04-11

Joined with Archie over Zoom (following his recovery from what's hopefully the worst of a bit of illness earlier this week) to coordinate per Assignment 4. The ideas we went over were mostly incremental refinements of ideas we'd discussed previously. Our major challenges will be:

1. __Time__. Specifically overlapping time; Archie and I (PEKO) need to coordinate to find perhaps 9 hours in total just for collecting samples. 3 to 4 sessions of data collection should be sufficient. Efficient collection and staying on task will be easier with both of us present, especially since we don't each have our own separate sensor setups.

2. __Procedure__. If we are shooting to collect 1000 events or so ourselves, we will need to be organized and better solidify our existing procedure. Even setting speed aside, keeping track of exactly what conditions we've collected will be critical for model building. So we can smoothly progress through the various states of interest, printing out a grid/checklist is probably easiest.

3. __Sampling robustness__. I (PEKO) wonder whether the Raspberry Pi that's on the breadboard we now have could be used to read the sensor at a high enough rate to be useful? This would make our setup a bit lack finicky.

**Number of samples:** As per my (PEKO's) submission for Assignment 4, hitting 500 to 1000 samples should not be out of reach if we are organized. We have about 5 locations we can use, with 3 or more plugs a piece. We have 3 distinct bulb-types and can try to see whether there is a major difference given wattage (20 vs 60, for incandescents) or things like frosting. Regardless, at least 3 types there. We can vary the distance of the sensor to the light source and call this "close/medium/far." Day/night conditions (or just general possible light interference) might be another multiplier (two states). We have a number of devices (6 or 7) that should not be overly difficult to transport or find in our test locations, which represent some different load categories/shapes; a vacuum cleaner should be an inductive load, as should a coffee grinder or a plug-in dremel. Resistive loads are easy to find (toasters, water kettles), which in some cases have a mixed element (a fan in a space heater or blow dryer). Nonlinear loads should include microwaves and televisions, though these will be hard to move so we won't be able to be as consistent about them. Although consumer devices are generally not capacative, an uninterrupted power supply might be an example of something we can find, but... to get back to counting matters, we have something like:

$$ nCr(~7 devices, 4 states including on-off order) \times 4 locations \times 3 distances \times 3 bulb types \times 3 plugs \times 2 natural-light states \times 2 sampling rates = 15,120 $$

Although capturing the full bredth of these combinations will not be possible, the above means there is plenty of space to explore.

Aside: One change not mentioned here previously is that we will have some participation on this project by another student in our academic program who unexpectedly showed interest; Vinitha may be interested in helping to collect samples, wrangling data or data piplines, ML models, or some combination. We absolutely must coordinate with her come Monday, as knowing exactly which tasks she's interested in will help with planning.
