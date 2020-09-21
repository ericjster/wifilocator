# wifilocator

This project is an investigation for finding the location (x,y,z coordinates) of a wifi signal source based on RSS (Relative Signal Strength) measured at different locations.

# Model

Some formulas from the literature:
    $RSSI = -10n log10(d) + A$
    $RSS = P * log( d / d0 ) + A$
But note that d vs. d/d0 is like changing units, do we care?

Simplifying:
    $RSS = c1 * log( d / c2 ) + c3$
    $(RSS-c3)/c1 = log( d / c2 )$
    $d = c2 * e^{(RSS-c3)/c1}$

So we have a formula to calculate distance based on RSS measurement. We can also calulate distance based on the xyz location of the source.

Therefore, we seek to minimize the MSE (Mean Squared Error):
    $sum( ( d_{rss} - d_{xyz} )^2 )$
Alternatively the L1 norm, or huber, or other loss function:
    $sum( huber( d_{rss} - d_{xyz} ) )$

# Sample Data

The directory "data" contains sample measurement using a Macbook Air and TP-Link AC1750 wifi router.

Here is a Desmos graph of the distance formula with parameters.
It is useful to try with the found values of c1, c2, c3.
(Remember that RSS is always negative, and large negative RSS correspond to large distances.)
https://www.desmos.com/calculator/eubsa5dfvu
