import time


def hms(seconds):
    h = int(seconds // 3600)
    m = int(seconds % 3600 // 60)
    s = int(seconds % 3600 % 60)
    #return '{:02f}:{:02f}:{:02f}'.format(h, m, s)
    return f'{h}:{m}:{s}'
hms(236.865)