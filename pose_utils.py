import numpy as np
from math import degrees

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arccos(np.clip(np.dot(a - b, c - b) /
                            (np.linalg.norm(a - b) * np.linalg.norm(c - b)), -1.0, 1.0))
    return degrees(rad)
