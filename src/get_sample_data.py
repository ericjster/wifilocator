"""
This file parses the sample data into a matrix.
"""

import re

DATAFILE = "../data/wifi_sample_1.txt"

def get_sample_data():
    """Return an array of data. 
    Each row is a sample.
    Columns: (x, y, z, RSS)"""

    rows = []
    xyz = None
    with open(DATAFILE) as f:
        for line in f:
            # if "location" in line:
            #     print(line)
            m = re.search("location.*\(([-0-9,]+)\)", line)
            if m:
                # print("Found:", m.group(1))
                xyz = m.group(1).split(",")
            m = re.search("MyMacAddress +([-0-9]+) ", line)
            if m:
                # print("Found:", m.group(1))
                rss = m.group(1)
                row = ( float(xyz[0]), float(xyz[1]), float(xyz[2]), float(rss) )
                rows.append(row)
    return rows

if __name__ == "__main__":
    rows = get_sample_data()
    print("Found:")
    for row in rows:
        print("  ", row)
