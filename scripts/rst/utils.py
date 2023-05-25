import os
import pandas as pd
import numpy as np

def hms(seconds):
    h = int(seconds // 3600)
    m = int(seconds % 3600 // 60)
    s = int(seconds % 3600 % 60)
    return f'{h}:{m}:{s}'