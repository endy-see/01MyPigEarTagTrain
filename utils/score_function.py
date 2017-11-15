import math
import os,sys


def thegama(distance, thredhold=1.0):
    assert thredhold > 0.5 and thredhold < 1.6
    # assert distance >= 0

    if distance < 0.01:
        return 100

    
    if distance <= thredhold:
        x = (thredhold - distance) / (thredhold - 0.5)
    else:
        x = (thredhold - distance) / (1.6 - thredhold)

    return int(100 / (1 + math.exp(-x)))


def line(distance, thredhold=1.0):
    assert thredhold > 0.5 and thredhold < 1.6
    assert distance >= 0

    if distance <= thredhold:
        return int(50 +  50 * (thredhold - distance) / (thredhold - 0.5))
    else:
        return max(0, int(50 - 50 * (distance - thredhold)/(1.6 - thredhold)))

def exp(distance, thredhold=1.0):
    assert thredhold > 0.5 and thredhold < 1.6
    if distance < 0.001:
        return 100

    x = (thredhold - distance) * 5.0
    return int(100 / (1 + math.exp(-x)))


for i in range(0, 200):
    x = i /100.0
    print("{}, {}, {}, {}".format(x, line(x, 1.12), thegama(x, 1.12), exp(x, 1.12)))