#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:09:17 2020

@author: matthewoshaughnessy
"""

import gif
from matplotlib import pyplot as plt


@gif.frame
def plot(x, y):
    plt.figure(figsize=(5, 3), dpi=100)
    plt.scatter(x, y)
    plt.xlim((0, 100))
    plt.ylim((0, 100))


from random import randint


frames = []
for _ in range(50):
    x = [randint(0, 100) for _ in range(10)]
    y = [randint(0, 100) for _ in range(10)]
    frame = plot(x, y)
    frames.append(frame)

    
gif.save(frames, "testgif.gif", duration=100)
print('done!')