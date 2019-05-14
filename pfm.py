#refer: https://github.com/Jackie-Chou/MC-CNN-python
#change on reference

import struct
import numpy as np

def readPfm(filename):
    f = open(filename, 'rb')
    line = f.readline()
    assert line.strip().decode(encoding="utf-8") == "Pf" # one sample per pixel
    line = f.readline()
    items = line.strip().split()
    width = int(items[0])
    height = int(items[1])
    line = f.readline()
    if float(line.strip()) < 0:  # little-endian
        fmt = "<f"
    else:
        fmt = ">f"
    maps = np.ndarray([height, width], dtype=np.float32)
    for h in range(height-1, -1, -1):
        for w in range(width):
            sample = f.read(4)
            maps[h, w], = struct.unpack("<f", sample)
    f.close()
    return maps

def writePfm(disparity_map, filename):
    assert len(disparity_map.shape) == 2
    height, width = disparity_map.shape
    disparity_map = disparity_map.astype(np.float32)
    o = open(filename, "wb")
    # header
    str_ = "Pf\n"
    o.write(str_.encode(encoding = "utf-8"))
    str_ = str(width)+" "+str(height)+"\n"
    o.write(str_.encode(encoding = "utf-8"))
    str_ = "-1.0\n"
    o.write(str_.encode(encoding = "utf-8"))
    # raster
    # NOTE: bottom up
    # little-endian
    fmt = "<f"
    for h in range(height-1, -1, -1):
        for w in range(width):
            o.write(struct.pack(fmt, disparity_map[h, w]))
    o.close()