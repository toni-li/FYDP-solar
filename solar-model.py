from mip import *
import math
import matplotlib.pyplot as plt
import numpy as np

# defining parameters
E0 = 2200000  # month's electricity usage (Wh) from user
month = 1 # electricity usage month from user 
heating = "electric" # dependent on user input electric or natural gas
postal_code = 'N4S' # first 3 digits of postal code

# seasonal electricity usage (Wh) with trend
# if heating is electric, summer demand is inflated by 30% and winter is inflated by 298%
if heating == "electric":
    if month == 0 or month == 1 or month == 11: # winter months
        E = [[(E0/2.98), (E0/2.98*(1.3)), (E0/2.98), E0]]
    elif month == 5 or month == 6 or month == 7: # summer months
        E = [[E0/1.3, E0, E0/1.3, E0/1.3*(2.98)]]
    else:
        E = [[E0, (E0*1.3), E0, (E0*2.98)]]

# if heating is natural gas, summer is inflated by 30% and fall, winter, summer are same
else:
    if month == 5 or month == 6 or month == 7: # summer months
        E = [[E0/1.3, E0, E0/1.3, E0/1.3]]
    else:
        E = [[E0, (E0*1.3), E0, E0]]

G = [[0.00017952, 0.00017952, 0.00017952, 0.00017952]] # cost of electricity from the grid at year 0($/Wh)
m = [[2.5,2.5,2.5,2.5]]  # yearly maintenance cost ($/panel)
B = 15000  # budget from user
C = 315*2.80  # cost of each solar panel ($/panel) (12 modules of 60cell)
Ap = 18.9  # area of solar panel (ft^2) (40 * 68 inches)
Ar = 1700  # area of the roof (ft^2) from user
Armax = (Ar / 2) * 0.8
P = 315 # capacity of each solar panel (W) per hour
F = 2500  # fixed costs of installing solar panels
d = []  # deterioration factor at year i (%)
T = 25  # lifespan of solar panels
S = 4 # 0, 1, 2, 3 = “Spring”, “Summer”, “Fall”, “Winter”
L = [92, 92, 91, 90] # number of days within each quarter

#filling in cost of electricity values (remain constant throughout seasons)
for t in range(1, T):
    yearly_cost = G[t - 1][0] + (G[t - 1][0] * 0.02)
    G.append([yearly_cost, yearly_cost, yearly_cost, yearly_cost])

# filling in depreciation values (remain constant throughout seasons)
for t in range(T):
    yearly_depreciation = (0.0007/4) * t
    d.append([yearly_depreciation, yearly_depreciation, yearly_depreciation, yearly_depreciation])

# function to fill in Et - linearly decreases by 1.03%
for t in range(1, T):
    yearly_decrease_spring = E[t - 1][0] - (E[t - 1][0] * 0.0103)
    yearly_decrease_summer = E[t - 1][1] - (E[t - 1][1] * 0.0103)
    yearly_decrease_fall = E[t - 1][2] - (E[t - 1][2] * 0.0103)
    yearly_decrease_winter = E[t - 1][3] - (E[t - 1][3] * 0.0103)
    E.append([yearly_decrease_spring, yearly_decrease_summer, yearly_decrease_fall, yearly_decrease_winter])

# convert m into present value (remain constant throughout seasons)
i = 0.00206
for t in range(1,T):
    quarterly_maintainence = (2.5/((1+i) ** t))
    m.append([quarterly_maintainence, quarterly_maintainence, quarterly_maintainence, quarterly_maintainence])

# order: [SH-Spring, SH-Summer, SH-Fall, SH-Winter, CF]
city_cf_sh = {'L1S': [591.5, 819.4, 473.3, 279.4, 15.3], 'L1T': [591.5, 819.4, 473.3, 279.4, 15.3], 'L1Z': [591.5, 819.4, 473.3, 279.4, 15.3], 'P0R': [591.5, 819.4, 473.3, 279.4, 14.8], 'L9K': [579.8, 846.3, 415.8, 272.9, 15.3], 'L9G': [579.8, 846.3, 415.8, 272.9, 15.3], 'L4M': [579.5, 738.9, 336.8, 272.6, 14.9], 'L4N': [579.5, 738.9, 336.8, 272.6, 14.9], 'K8R': [524.7, 760.7, 394.5, 279.4, 15.4], 'K8P': [524.7, 760.7, 394.5, 279.4, 15.4], 'K8N': [524.7, 760.7, 394.5, 279.4, 15.4], 'L1C': [567.5, 784.9, 440.0, 276.3, 15.4], 'L1B': [567.5, 784.9, 440.0, 276.3, 15.4], 'L6Z': [591.5, 819.4, 473.3, 279.4, 15.5], 'L6X': [591.5, 819.4, 473.3, 279.4, 15.5], 'L6T': [591.5, 819.4, 473.3, 279.4, 15.5], 'L6S': [591.5, 819.4, 473.3, 279.4, 15.5], 'L6V': [591.5, 819.4, 473.3, 279.4, 15.5], 'L6Y': [591.5, 819.4, 473.3, 279.4, 15.5], 'L6R': [591.5, 819.4, 473.3, 279.4, 15.5], 'L6P': [591.5, 819.4, 473.3, 279.4, 15.5], 'L6W': [591.5, 819.4, 473.3, 279.4, 15.5], 'N3P': [579.8, 846.3, 415.8, 272.9, 15.2], 'N3S': [579.8, 846.3, 415.8, 272.9, 15.2], 'N3T': [579.8, 846.3, 415.8, 272.9, 15.2], 'N3V': [579.8, 846.3, 415.8, 272.9, 15.2], 'N3R': [579.8, 846.3, 415.8, 272.9, 15.2], 'L7P': [579.8, 846.3, 415.8, 272.9, 15.5], 'L7T': [579.8, 846.3, 415.8, 272.9, 15.5], 'L7N': [579.8, 846.3, 415.8, 272.9, 15.5], 'L7S': [579.8, 846.3, 415.8, 272.9, 15.5], 'L7L': [579.8, 846.3, 415.8, 272.9, 15.5], 'L7R': [579.8, 846.3, 415.8, 272.9, 15.5], 'L7M': [579.8, 846.3, 415.8, 272.9, 15.5], 'N3E': [549.1, 772.6, 391.5, 234.4, 14.9], 'N1S': [549.1, 772.6, 391.5, 234.4, 14.9], 'N3H': [549.1, 772.6, 391.5, 234.4, 14.9], 'N3C': [549.1, 772.6, 391.5, 234.4, 14.9], 'N1P': [549.1, 772.6, 391.5, 234.4, 14.9], 'N1R': [549.1, 772.6, 391.5, 234.4, 14.9], 'N1T': [549.1, 772.6, 391.5, 234.4, 14.9], 'N7M': [503.1, 720.4, 361.2, 207.4, 15.5], 'N7L': [503.1, 720.4, 361.2, 207.4, 15.5], 'K6K': [567.2, 806.2, 454.8, 303.9, 14.5], 'K6H': [567.2, 806.2, 454.8, 303.9, 14.5], 'K6J': [567.2, 806.2, 454.8, 303.9, 14.5], 'M3B': [591.5, 819.4, 473.3, 279.4, 15.6], 'M3C': [591.5, 819.4, 473.3, 279.4, 15.6], 'M3M': [591.5, 819.4, 473.3, 279.4, 15.6], 'M3L': [591.5, 819.4, 473.3, 279.4, 15.6], 'L0N': [583.0, 797.0, 425.0, 267.0, 15.2], 'M4G': [591.5, 819.4, 473.3, 279.4, 15.3], 'M4B': [591.5, 819.4, 473.3, 279.4, 15.3], 'M4H': [591.5, 819.4, 473.3, 279.4, 15.3], 'M4C': [591.5, 819.4, 473.3, 279.4, 15.3], 'M9R': [591.5, 819.4, 473.3, 279.4, 15.8], 'M9W': [591.5, 819.4, 473.3, 279.4, 15.8], 'M8Z': [591.5, 819.4, 473.3, 279.4, 15.8], 'M8W': [591.5, 819.4, 473.3, 279.4, 15.8], 'M8Y': [591.5, 819.4, 473.3, 279.4, 15.8], 'M9B': [591.5, 819.4, 473.3, 279.4, 15.8], 'M9V': [591.5, 819.4, 473.3, 279.4, 15.8], 'M8X': [591.5, 819.4, 473.3, 279.4, 15.8], 'M9A': [591.5, 819.4, 473.3, 279.4, 15.8], 'M9P': [591.5, 819.4, 473.3, 279.4, 15.8], 'M9C': [591.5, 819.4, 473.3, 279.4, 15.8], 'K0H': [524.7, 760.7, 394.5, 279.4, 14.9], 'K1B': [567.2, 806.2, 454.8, 303.9, 15.2], 'K1X': [567.2, 806.2, 454.8, 303.9, 15.2], 'K1C': [567.2, 806.2, 454.8, 303.9, 15.2], 'K1W': [567.2, 806.2, 454.8, 303.9, 15.2], 'K1T': [567.2, 806.2, 454.8, 303.9, 15.2], 'K1J': [567.2, 806.2, 454.8, 303.9, 15.2], 'P3B': [552.0, 767.0, 334.0, 267.0, 14.7], 'P3E': [552.0, 767.0, 334.0, 267.0, 14.7], 'P3C': [552.0, 767.0, 334.0, 267.0, 14.7], 'P3A': [552.0, 767.0, 334.0, 267.0, 14.7], 'P3Y': [552.0, 767.0, 334.0, 267.0, 14.7], 'P3L': [552.0, 767.0, 334.0, 267.0, 14.7], 'P3P': [552.0, 767.0, 334.0, 267.0, 14.7], 'P3N': [552.0, 767.0, 334.0, 267.0, 14.7], 'P3G': [552.0, 767.0, 334.0, 267.0, 14.7], 'N1E': [549.1, 772.6, 391.5, 234.4, 15.4], 'N1G': [549.1, 772.6, 391.5, 234.4, 15.4], 'N1L': [549.1, 772.6, 391.5, 234.4, 15.4], 'N1K': [549.1, 772.6, 391.5, 234.4, 15.4], 'N1H': [549.1, 772.6, 391.5, 234.4, 15.4], 'N1C': [549.1, 772.6, 391.5, 234.4, 15.4], 'L0P': [591.5, 819.4, 473.3, 279.4, 15.4], 'L9E': [591.5, 819.4, 473.3, 279.4, 15.4], 'L9B': [579.8, 846.3, 415.8, 272.9, 15.4], 'L8N': [579.8, 846.3, 415.8, 272.9, 15.4], 'L8J': [579.8, 846.3, 415.8, 272.9, 15.4], 'L8W': [579.8, 846.3, 415.8, 272.9, 15.4], 'L8P': [579.8, 846.3, 415.8, 272.9, 15.4], 'L8H': [579.8, 846.3, 415.8, 272.9, 15.4], 'L9C': [579.8, 846.3, 415.8, 272.9, 15.4], 'L8T': [579.8, 846.3, 415.8, 272.9, 15.4], 'L8S': [579.8, 846.3, 415.8, 272.9, 15.4], 'L8G': [579.8, 846.3, 415.8, 272.9, 15.4], 'L8R': [579.8, 846.3, 415.8, 272.9, 15.4], 'L8M': [579.8, 846.3, 415.8, 272.9, 15.4], 'L8V': [579.8, 846.3, 415.8, 272.9, 15.4], 'L8E': [579.8, 846.3, 415.8, 272.9, 15.4], 'L8K': [579.8, 846.3, 415.8, 272.9, 15.4], 'L9A': [579.8, 846.3, 415.8, 272.9, 15.4], 'L8L': [579.8, 846.3, 415.8, 272.9, 15.4], 'K2W': [567.2, 806.2, 454.8, 303.9, 14.7], 'K2K': [567.2, 806.2, 454.8, 303.9, 14.7], 'K2L': [567.2, 806.2, 454.8, 303.9, 14.7], 'K2V': [567.2, 806.2, 454.8, 303.9, 14.7], 'K2M': [567.2, 806.2, 454.8, 303.9, 14.7], 'K2T': [567.2, 806.2, 454.8, 303.9, 14.7], 'K7K': [524.7, 760.7, 394.5, 279.4, 15.4], 'K7L': [524.7, 760.7, 394.5, 279.4, 15.4], 'K7P': [524.7, 760.7, 394.5, 279.4, 15.4], 'K7M': [524.7, 760.7, 394.5, 279.4, 15.4], 'N2B': [549.1, 772.6, 391.5, 234.4, 14.6], 'N2P': [549.1, 772.6, 391.5, 234.4, 14.6], 'N2G': [549.1, 772.6, 391.5, 234.4, 14.6], 'N2A': [549.1, 772.6, 391.5, 234.4, 14.6], 'N2H': [549.1, 772.6, 391.5, 234.4, 14.6], 'N2R': [549.1, 772.6, 391.5, 234.4, 14.6], 'N2K': [549.1, 772.6, 391.5, 234.4, 14.6], 'N2E': [549.1, 772.6, 391.5, 234.4, 14.6], 'N2C': [549.1, 772.6, 391.5, 234.4, 14.6], 'N2M': [549.1, 772.6, 391.5, 234.4, 14.6], 'N2N': [549.1, 772.6, 391.5, 234.4, 14.6], 'N9H': [579.8, 846.3, 415.8, 272.9, 14.6], 'N9J': [579.8, 846.3, 415.8, 272.9, 14.6], 'L0L': [579.5, 738.9, 336.8, 272.6, 14.8], 'L0K': [579.5, 738.9, 336.8, 272.6, 14.8], 'L0E': [579.5, 738.9, 336.8, 272.6, 14.8], 'P0T': [634.6, 784.9, 376.0, 324.4, 14.8], 'P0S': [634.6, 784.9, 376.0, 324.4, 14.8], 'N6H': [503.1, 720.4, 361.2, 207.4, 14.5], 'N6N': [503.1, 720.4, 361.2, 207.4, 14.5], 'N5X': [503.1, 720.4, 361.2, 207.4, 14.5], 'N6K': [503.1, 720.4, 361.2, 207.4, 14.5], 'N6J': [503.1, 720.4, 361.2, 207.4, 14.5], 'N5Y': [503.1, 720.4, 361.2, 207.4, 14.5], 'N6A': [503.1, 720.4, 361.2, 207.4, 14.5], 'N5Z': [503.1, 720.4, 361.2, 207.4, 14.5], 'N6B': [503.1, 720.4, 361.2, 207.4, 14.5], 'N5V': [503.1, 720.4, 361.2, 207.4, 14.5], 'N6P': [503.1, 720.4, 361.2, 207.4, 14.5], 'N6G': [503.1, 720.4, 361.2, 207.4, 14.5], 'N6L': [503.1, 720.4, 361.2, 207.4, 14.5], 'N6E': [503.1, 720.4, 361.2, 207.4, 14.5], 'N6C': [503.1, 720.4, 361.2, 207.4, 14.5], 'N5W': [503.1, 720.4, 361.2, 207.4, 14.5], 'N6M': [503.1, 720.4, 361.2, 207.4, 14.5], 'L6G': [591.5, 819.4, 473.3, 279.4, 15.1], 'L6C': [591.5, 819.4, 473.3, 279.4, 15.1], 'L3R': [591.5, 819.4, 473.3, 279.4, 15.1], 'L6E': [591.5, 819.4, 473.3, 279.4, 15.1], 'L3S': [591.5, 819.4, 473.3, 279.4, 15.1], 'L6B': [591.5, 819.4, 473.3, 279.4, 15.1], 'L3P': [591.5, 819.4, 473.3, 279.4, 15.1], 'L5B': [591.5, 819.4, 473.3, 279.4, 15.2], 'L4Z': [591.5, 819.4, 473.3, 279.4, 15.2], 'L4Y': [591.5, 819.4, 473.3, 279.4, 15.2], 'L5A': [591.5, 819.4, 473.3, 279.4, 15.2], 'L5H': [591.5, 819.4, 473.3, 279.4, 15.2], 'L5R': [591.5, 819.4, 473.3, 279.4, 15.2], 'L5E': [591.5, 819.4, 473.3, 279.4, 15.2], 'L5J': [591.5, 819.4, 473.3, 279.4, 15.2], 'L5T': [591.5, 819.4, 473.3, 279.4, 15.2], 'L4T': [591.5, 819.4, 473.3, 279.4, 15.2], 'L5M': [591.5, 819.4, 473.3, 279.4, 15.2], 'L4V': [591.5, 819.4, 473.3, 279.4, 15.2], 'L5C': [591.5, 819.4, 473.3, 279.4, 15.2], 'L5V': [591.5, 819.4, 473.3, 279.4, 15.2], 'L5W': [591.5, 819.4, 473.3, 279.4, 15.2], 'L4X': [591.5, 819.4, 473.3, 279.4, 15.2], 'L5S': [591.5, 819.4, 473.3, 279.4, 15.2], 'L5P': [591.5, 819.4, 473.3, 279.4, 15.2], 'L4W': [591.5, 819.4, 473.3, 279.4, 15.2], 'L5G': [591.5, 819.4, 473.3, 279.4, 15.2], 'L5K': [591.5, 819.4, 473.3, 279.4, 15.2], 'L5L': [591.5, 819.4, 473.3, 279.4, 15.2], 'L5N': [591.5, 819.4, 473.3, 279.4, 15.2], 'K2G': [567.2, 806.2, 454.8, 303.9, 15.2], 'K2R': [567.2, 806.2, 454.8, 303.9, 15.2], 'K2E': [567.2, 806.2, 454.8, 303.9, 15.2], 'K2H': [567.2, 806.2, 454.8, 303.9, 15.2], 'K2J': [567.2, 806.2, 454.8, 303.9, 15.2], 'L3X': [591.5, 819.4, 473.3, 279.4, 15.2], 'L3Y': [591.5, 819.4, 473.3, 279.4, 15.2], 'L2G': [583.0, 797.0, 425.0, 267.0, 15.2], 'L2H': [583.0, 797.0, 425.0, 267.0, 15.2], 'L2E': [583.0, 797.0, 425.0, 267.0, 15.2], 'L2J': [583.0, 797.0, 425.0, 267.0, 15.2], 'P0B': [579.5, 738.9, 336.8, 272.6, 15.3], 'P0H': [579.5, 738.9, 336.8, 272.6, 15.3], 'P0A': [579.5, 738.9, 336.8, 272.6, 15.3], 'P1B': [579.5, 738.9, 336.8, 272.6, 15.3], 'P1C': [579.5, 738.9, 336.8, 272.6, 15.3], 'P1A': [579.5, 738.9, 336.8, 272.6, 15.3], 'M6A': [567.5, 784.9, 440.0, 276.3, 15.2], 'M2K': [567.5, 784.9, 440.0, 276.3, 15.2], 'M3A': [567.5, 784.9, 440.0, 276.3, 15.2], 'M9L': [567.5, 784.9, 440.0, 276.3, 15.2], 'M3H': [567.5, 784.9, 440.0, 276.3, 15.2], 'M3J': [567.5, 784.9, 440.0, 276.3, 15.2], 'M9M': [567.5, 784.9, 440.0, 276.3, 15.2], 'M2J': [567.5, 784.9, 440.0, 276.3, 15.2], 'M6L': [567.5, 784.9, 440.0, 276.3, 15.2], 'M2P': [567.5, 784.9, 440.0, 276.3, 15.2], 'M2H': [567.5, 784.9, 440.0, 276.3, 15.2], 'M2L': [567.5, 784.9, 440.0, 276.3, 15.2], 'M6B': [567.5, 784.9, 440.0, 276.3, 15.2], 'M3N': [567.5, 784.9, 440.0, 276.3, 15.2], 'L6H': [591.5, 819.4, 473.3, 279.4, 15.3], 'L6M': [591.5, 819.4, 473.3, 279.4, 15.3], 'L6L': [591.5, 819.4, 473.3, 279.4, 15.3], 'L6K': [591.5, 819.4, 473.3, 279.4, 15.3], 'L6J': [591.5, 819.4, 473.3, 279.4, 15.3], 'L9W': [591.5, 819.4, 473.3, 279.4, 15.3], 'L9V': [591.5, 819.4, 473.3, 279.4, 15.3], 'L1H': [591.5, 819.4, 473.3, 279.4, 14.4], 'L1K': [591.5, 819.4, 473.3, 279.4, 14.4], 'L1J': [591.5, 819.4, 473.3, 279.4, 14.4], 'L1L': [591.5, 819.4, 473.3, 279.4, 14.4], 'L1G': [591.5, 819.4, 473.3, 279.4, 14.4], 'K1L': [567.2, 806.2, 454.8, 303.9, 14.4], 'K1K': [567.2, 806.2, 454.8, 303.9, 14.4], 'K2A': [567.2, 806.2, 454.8, 303.9, 14.4], 'K1R': [567.2, 806.2, 454.8, 303.9, 14.4], 'K1Y': [567.2, 806.2, 454.8, 303.9, 14.4], 'K1V': [567.2, 806.2, 454.8, 303.9, 14.4], 'K1G': [567.2, 806.2, 454.8, 303.9, 14.4], 'K2P': [567.2, 806.2, 454.8, 303.9, 14.4], 'K1A': [567.2, 806.2, 454.8, 303.9, 14.4], 'K1Z': [567.2, 806.2, 454.8, 303.9, 14.4], 'K2C': [567.2, 806.2, 454.8, 303.9, 14.4], 'K1N': [567.2, 806.2, 454.8, 303.9, 14.4], 'K1H': [567.2, 806.2, 454.8, 303.9, 14.4], 'K1M': [567.2, 806.2, 454.8, 303.9, 14.4], 'K2B': [567.2, 806.2, 454.8, 303.9, 14.4], 'K1S': [567.2, 806.2, 454.8, 303.9, 14.4], 'K1P': [567.2, 806.2, 454.8, 303.9, 14.4], 'P0G': [579.5, 738.9, 336.8, 272.6, 15.0], 'P0E': [579.5, 738.9, 336.8, 272.6, 15.0], 'P0C': [579.5, 738.9, 336.8, 272.6, 15.0], 'K8B': [567.2, 806.2, 454.8, 303.9, 15.0], 'K8A': [567.2, 806.2, 454.8, 303.9, 15.0], 'K9L': [536.6, 809.5, 382.3, 269.8, 14.6], 'K9H': [536.6, 809.5, 382.3, 269.8, 14.6], 'K9J': [536.6, 809.5, 382.3, 269.8, 14.6], 'K9K': [536.6, 809.5, 382.3, 269.8, 14.6], 'K0L': [536.6, 809.5, 382.3, 269.8, 14.6], 'L1X': [591.5, 819.4, 473.3, 279.4, 14.6], 'L1Y': [591.5, 819.4, 473.3, 279.4, 14.6], 'L1V': [591.5, 819.4, 473.3, 279.4, 14.6], 'L1W': [591.5, 819.4, 473.3, 279.4, 14.6], 'K0J': [567.2, 806.2, 454.8, 303.9, 14.8], 'L4S': [567.5, 784.9, 440.0, 276.3, 14.8], 'L4B': [567.5, 784.9, 440.0, 276.3, 14.8], 'L4E': [567.5, 784.9, 440.0, 276.3, 14.8], 'L4C': [567.5, 784.9, 440.0, 276.3, 14.8], 'N7W': [561.1, 818.8, 430.9, 249.6, 14.6], 'N7T': [561.1, 818.8, 430.9, 249.6, 14.6], 'N7X': [561.1, 818.8, 430.9, 249.6, 14.6], 'N7S': [561.1, 818.8, 430.9, 249.6, 14.6], 'N7V': [561.1, 818.8, 430.9, 249.6, 14.6], 'P6C': [644.0, 797.0, 334.0, 295.0, 14.6], 'P6B': [644.0, 797.0, 334.0, 295.0, 14.6], 'P6A': [644.0, 797.0, 334.0, 295.0, 14.6], 'M1J': [567.5, 784.9, 440.0, 276.3, 14.6], 'M1N': [567.5, 784.9, 440.0, 276.3, 14.6], 'M1X': [567.5, 784.9, 440.0, 276.3, 14.6], 'M1V': [567.5, 784.9, 440.0, 276.3, 14.6], 'M1R': [567.5, 784.9, 440.0, 276.3, 14.6], 'M1B': [567.5, 784.9, 440.0, 276.3, 14.6], 'M1L': [567.5, 784.9, 440.0, 276.3, 14.6], 'M1P': [567.5, 784.9, 440.0, 276.3, 14.6], 'M1E': [567.5, 784.9, 440.0, 276.3, 14.6], 'M1S': [567.5, 784.9, 440.0, 276.3, 14.6], 'M1C': [567.5, 784.9, 440.0, 276.3, 14.6], 'M1T': [567.5, 784.9, 440.0, 276.3, 14.6], 'M1K': [567.5, 784.9, 440.0, 276.3, 14.6], 'M1M': [567.5, 784.9, 440.0, 276.3, 14.6], 'M1W': [567.5, 784.9, 440.0, 276.3, 14.6], 'M1H': [567.5, 784.9, 440.0, 276.3, 14.6], 'M1G': [567.5, 784.9, 440.0, 276.3, 14.6], 'L2W': [549.3, 812.2, 422.0, 239.7, 16.1], 'L2R': [549.3, 812.2, 422.0, 239.7, 16.1], 'L2S': [549.3, 812.2, 422.0, 239.7, 16.1], 'L2T': [549.3, 812.2, 422.0, 239.7, 16.1], 'L2P': [549.3, 812.2, 422.0, 239.7, 16.1], 'L2V': [549.3, 812.2, 422.0, 239.7, 16.1], 'L2M': [549.3, 812.2, 422.0, 239.7, 16.1], 'L2N': [549.3, 812.2, 422.0, 239.7, 16.1], 'N5P': [503.1, 720.4, 361.2, 207.4, 14.6], 'N5R': [503.1, 720.4, 361.2, 207.4, 14.6], 'N8V': [503.1, 720.4, 361.2, 207.4, 14.7], 'N8N': [503.1, 720.4, 361.2, 207.4, 14.7], 'P7A': [634.6, 784.9, 376.0, 324.4, 12.2], 'P7G': [634.6, 784.9, 376.0, 324.4, 12.2], 'P7C': [634.6, 784.9, 376.0, 324.4, 12.2], 'P7B': [634.6, 784.9, 376.0, 324.4, 12.2], 'P7E': [634.6, 784.9, 376.0, 324.4, 12.2], 'P7K': [634.6, 784.9, 376.0, 324.4, 12.2], 'P7J': [634.6, 784.9, 376.0, 324.4, 12.2], 'P0K': [583.0, 705.0, 303.0, 267.0, 15.1], 'P0J': [583.0, 705.0, 303.0, 267.0, 15.1], 'P4P': [583.0, 705.0, 303.0, 267.0, 15.1], 'P4N': [583.0, 705.0, 303.0, 267.0, 15.1], 'P4R': [583.0, 705.0, 303.0, 267.0, 15.1], 'M5W': [567.5, 784.9, 440.0, 276.3, 14.2], 'M4E': [567.5, 784.9, 440.0, 276.3, 14.2], 'M6K': [567.5, 784.9, 440.0, 276.3, 14.2], 'M5N': [567.5, 784.9, 440.0, 276.3, 14.2], 'M4T': [567.5, 784.9, 440.0, 276.3, 14.2], 'M4V': [567.5, 784.9, 440.0, 276.3, 14.2], 'M4S': [567.5, 784.9, 440.0, 276.3, 14.2], 'M5T': [567.5, 784.9, 440.0, 276.3, 14.2], 'M5B': [567.5, 784.9, 440.0, 276.3, 14.2], 'M6P': [567.5, 784.9, 440.0, 276.3, 14.2], 'M4L': [567.5, 784.9, 440.0, 276.3, 14.2], 'M4Y': [567.5, 784.9, 440.0, 276.3, 14.2], 'M6S': [567.5, 784.9, 440.0, 276.3, 14.2], 'M5X': [567.5, 784.9, 440.0, 276.3, 14.2], 'M6G': [567.5, 784.9, 440.0, 276.3, 14.2], 'M5J': [567.5, 784.9, 440.0, 276.3, 14.2], 'M5C': [567.5, 784.9, 440.0, 276.3, 14.2], 'M5S': [567.5, 784.9, 440.0, 276.3, 14.2], 'M5A': [567.5, 784.9, 440.0, 276.3, 14.2], 'M4K': [567.5, 784.9, 440.0, 276.3, 14.2], 'M5L': [567.5, 784.9, 440.0, 276.3, 14.2], 'M6R': [567.5, 784.9, 440.0, 276.3, 14.2], 'M5K': [567.5, 784.9, 440.0, 276.3, 14.2], 'M7Y': [567.5, 784.9, 440.0, 276.3, 14.2], 'M6H': [567.5, 784.9, 440.0, 276.3, 14.2], 'M5G': [567.5, 784.9, 440.0, 276.3, 14.2], 'M5R': [567.5, 784.9, 440.0, 276.3, 14.2], 'M6J': [567.5, 784.9, 440.0, 276.3, 14.2], 'M4X': [567.5, 784.9, 440.0, 276.3, 14.2], 'M4R': [567.5, 784.9, 440.0, 276.3, 14.2], 'M4P': [567.5, 784.9, 440.0, 276.3, 14.2], 'M4J': [567.5, 784.9, 440.0, 276.3, 14.2], 'M5E': [567.5, 784.9, 440.0, 276.3, 14.2], 'M3K': [567.5, 784.9, 440.0, 276.3, 14.2], 'M5V': [567.5, 784.9, 440.0, 276.3, 14.2], 'M4N': [567.5, 784.9, 440.0, 276.3, 14.2], 'M4M': [567.5, 784.9, 440.0, 276.3, 14.2], 'M4W': [567.5, 784.9, 440.0, 276.3, 14.2], 'M5P': [567.5, 784.9, 440.0, 276.3, 14.2], 'M5H': [567.5, 784.9, 440.0, 276.3, 14.2], 'M8V': [567.5, 784.9, 440.0, 276.3, 14.2], 'N2V': [549.1, 772.6, 391.5, 234.4, 15.4], 'N2J': [549.1, 772.6, 391.5, 234.4, 15.4], 'N2L': [549.1, 772.6, 391.5, 234.4, 15.4], 'N2T': [549.1, 772.6, 391.5, 234.4, 15.4], 'L3C': [549.1, 772.6, 391.5, 234.4, 15.4], 'L3B': [567.5, 784.9, 440.0, 276.3, 15.4], 'L1N': [567.5, 784.9, 440.0, 276.3, 14.6], 'L1R': [567.5, 784.9, 440.0, 276.3, 14.6], 'L1P': [567.5, 784.9, 440.0, 276.3, 14.6], 'L1M': [567.5, 784.9, 440.0, 276.3, 14.6], 'M2R': [567.5, 784.9, 440.0, 276.3, 14.6], 'M2M': [567.5, 784.9, 440.0, 276.3, 14.6], 'M2N': [567.5, 784.9, 440.0, 276.3, 14.6], 'N8S': [503.1, 720.4, 361.2, 207.4, 14.6], 'N9E': [503.1, 720.4, 361.2, 207.4, 14.6], 'N8P': [503.1, 720.4, 361.2, 207.4, 14.6], 'N8Y': [503.1, 720.4, 361.2, 207.4, 14.6], 'N8W': [503.1, 720.4, 361.2, 207.4, 14.6], 'N8X': [503.1, 720.4, 361.2, 207.4, 14.6], 'N9B': [503.1, 720.4, 361.2, 207.4, 14.6], 'N8R': [503.1, 720.4, 361.2, 207.4, 14.6], 'N9G': [503.1, 720.4, 361.2, 207.4, 14.6], 'N9C': [503.1, 720.4, 361.2, 207.4, 14.6], 'N8T': [503.1, 720.4, 361.2, 207.4, 14.6], 'N9A': [503.1, 720.4, 361.2, 207.4, 14.6], 'L4H': [567.5, 784.9, 440.0, 276.3, 14.6], 'L4L': [567.5, 784.9, 440.0, 276.3, 14.6], 'N4T': [503.1, 720.4, 361.2, 207.4, 14.6], 'N4S': [503.1, 720.4, 361.2, 207.4, 14.6], 'N4V': [503.1, 720.4, 361.2, 207.4, 14.6], 'M6N': [567.5, 784.9, 440.0, 276.3, 14.6], 'M6E': [567.5, 784.9, 440.0, 276.3, 14.6], 'M6M': [567.5, 784.9, 440.0, 276.3, 14.6], 'M6C': [567.5, 784.9, 440.0, 276.3, 14.6], 'L7J': [567.5, 784.9, 440.0, 276.3, 15.2], 'N9V': [503.1, 720.4, 361.2, 207.4, 14.8], 'L4G': [591.5, 819.4, 473.3, 279.4, 15.3], 'P0M': [552.0, 767.0, 364.0, 298.0, 14.8], 'K7N': [503.1, 720.4, 361.2, 207.4, 14.8], 'N5H': [503.1, 720.4, 361.2, 207.4, 15.3], 'L9R': [591.5, 819.4, 473.3, 279.4, 14.8], 'K7S': [567.2, 806.2, 454.8, 303.9, 15.3], 'N3A': [549.1, 772.6, 391.5, 234.4, 15.3], 'L9J': [579.5, 738.9, 336.8, 272.6, 14.9], 'L9X': [579.5, 738.9, 336.8, 272.6, 14.9], 'L7E': [591.5, 819.4, 473.3, 279.4, 15.4], 'L7A': [591.5, 819.4, 473.3, 279.4, 15.5], 'N0H': [579.5, 738.9, 336.8, 272.6, 15.2], 'P1L': [579.5, 738.9, 336.8, 272.6, 15.4], 'N0E': [579.8, 846.3, 415.8, 272.9, 15.2], 'L3Z': [591.5, 819.4, 473.3, 279.4, 15.4], 'K6V': [524.7, 760.7, 394.5, 279.4, 15.2], 'L7C': [591.5, 819.4, 473.3, 279.4, 15.5], 'K7C': [567.2, 806.2, 454.8, 303.9, 14.9], 'L9Y': [579.5, 738.9, 336.8, 272.6, 15.5], 'K4C': [567.2, 806.2, 454.8, 303.9, 14.5], 'L7K': [591.5, 819.4, 473.3, 279.4, 15.5], 'K9A': [524.7, 760.7, 394.5, 279.4, 15.5], 'L4K': [503.1, 720.4, 361.2, 207.4, 15.5], 'K4B': [567.2, 806.2, 454.8, 303.9, 14.5], 'N3W': [579.8, 846.3, 415.8, 272.9, 15.5], 'P0L': [583.0, 705.0, 303.0, 267.0, 15.5], 'L1E': [591.5, 819.4, 473.3, 279.4, 14.5], 'N4B': [503.1, 720.4, 361.2, 207.4, 14.5], 'N1A': [583.0, 797.0, 425.0, 267.0, 15.2], 'P8N': [634.6, 784.9, 376.0, 324.4, 15.6], 'L9H': [591.5, 819.4, 473.3, 279.4, 15.2], 'L0B': [591.5, 819.4, 473.3, 279.4, 15.2], 'K6T': [524.7, 760.7, 394.5, 279.4, 14.8], 'P5E': [552.0, 767.0, 334.0, 267.0, 15.1], 'L0R': [579.8, 846.3, 415.8, 272.9, 15.2], 'P5A': [552.0, 767.0, 334.0, 267.0, 14.8], 'N0L': [503.1, 720.4, 361.2, 207.4, 15.1], 'N3B': [549.1, 772.6, 391.5, 234.4, 14.7], 'N0R': [503.1, 720.4, 361.2, 207.4, 15.8], 'N8M': [503.1, 720.4, 361.2, 207.4, 15.8], 'N1M': [549.1, 772.6, 391.5, 234.4, 15.1], 'L2A': [583.0, 797.0, 425.0, 267.0, 14.8], 'P9A': [634.6, 784.9, 376.0, 324.4, 14.9], 'K7G': [524.7, 760.7, 394.5, 279.4, 14.4], 'N0C': [552.0, 767.0, 334.0, 267.0, 15.2], 'L7G': [591.5, 819.4, 473.3, 279.4, 15.5], 'N7A': [503.1, 720.4, 361.2, 207.4, 14.5], 'L3M': [579.8, 846.3, 415.8, 272.9, 15.4], 'L0M': [552.0, 767.0, 334.0, 267.0, 15.2], 'P1P': [579.5, 738.9, 336.8, 272.6, 14.7], 'N4N': [549.1, 772.6, 391.5, 234.4, 14.6], 'P1H': [579.5, 738.9, 336.8, 272.6, 14.6], 'K6A': [567.2, 806.2, 454.8, 303.9, 14.6], 'N0G': [503.1, 720.4, 361.2, 207.4, 14.5], 'L9N': [591.5, 819.4, 473.3, 279.4, 15.4], 'N5C': [503.1, 720.4, 361.2, 207.4, 15.0], 'L9S': [579.5, 738.9, 336.8, 272.6, 14.7], 'P5N': [583.0, 705.0, 303.0, 267.0, 14.3], 'P0X': [591.5, 819.4, 473.3, 279.4, 14.7], 'N2Z': [549.1, 772.6, 391.5, 234.4, 14.6], 'P2N': [552.0, 767.0, 334.0, 267.0, 14.6], 'K0M': [536.6, 809.5, 382.3, 269.8, 15.2], 'N0P': [503.1, 720.4, 361.2, 207.4, 14.6], 'L7B': [591.5, 819.4, 473.3, 279.4, 15.4], 'P9N': [634.6, 784.9, 376.0, 324.4, 15.1], 'L4P': [591.5, 819.4, 473.3, 279.4, 15.2], 'N9Y': [503.1, 720.4, 361.2, 207.4, 15.5], 'P0Y': [634.6, 784.9, 376.0, 324.4, 14.8], 'K9V': [536.6, 809.5, 382.3, 269.8, 15.2], 'N0N': [561.1, 818.8, 430.9, 249.6, 15.1], 'N4W': [549.1, 772.6, 391.5, 234.4, 14.5], 'N8H': [503.1, 720.4, 361.2, 207.4, 15.6], 'P0P': [579.5, 738.9, 336.8, 272.6, 15.3], 'N4L': [579.5, 738.9, 336.8, 272.6, 15.1], 'L4R': [579.5, 738.9, 336.8, 272.6, 15.2], 'K4M': [567.2, 806.2, 454.8, 303.9, 14.5], 'N0M': [503.1, 720.4, 361.2, 207.4, 14.9], 'L9T': [567.5, 784.9, 440.0, 276.3, 15.2], 'L6A': [567.5, 784.9, 440.0, 276.3, 15.1], 'K7R': [524.7, 760.7, 394.5, 279.4, 14.8], 'L0S': [583.0, 797.0, 425.0, 267.0, 15.3], 'K0A': [567.2, 806.2, 454.8, 303.9, 15.2], 'L0J': [591.5, 819.4, 473.3, 279.4, 15.3], 'P7L': [634.6, 784.9, 376.0, 324.4, 15.2], 'M4A': [567.5, 784.9, 440.0, 276.3, 15.2], 'M5M': [567.5, 784.9, 440.0, 276.3, 15.2], 'P0V': [579.5, 738.9, 336.8, 272.6, 15.3], 'L0G': [579.5, 738.9, 336.8, 272.6, 15.3], 'L3V': [579.5, 738.9, 336.8, 272.6, 14.6], 'K1E': [567.2, 806.2, 454.8, 303.9, 14.4], 'K4A': [567.2, 806.2, 454.8, 303.9, 14.4], 'N4K': [579.5, 738.9, 336.8, 272.6, 15.0], 'N0J': [503.1, 720.4, 361.2, 207.4, 14.9], 'N3L': [549.1, 772.6, 391.5, 234.4, 15.1], 'K7H': [579.5, 738.9, 336.8, 272.6, 14.8], 'N0K': [579.5, 738.9, 336.8, 272.6, 14.8], 'K8H': [555.1, 763.7, 336.9, 284.1, 14.6], 'L9L': [567.5, 784.9, 440.0, 276.3, 15.2], 'P2A': [579.5, 738.9, 336.8, 272.6, 15.0], 'L3K': [583.0, 797.0, 425.0, 267.0, 15.1], 'N5L': [503.1, 720.4, 361.2, 207.4, 14.7], 'L9M': [579.5, 738.9, 336.8, 272.6, 15.3], 'L1A': [536.6, 809.5, 382.3, 269.8, 15.3], 'K0B': [567.2, 806.2, 454.8, 303.9, 14.7], 'M7A': [567.5, 784.9, 440.0, 276.3, 15.2], 'K0K': [536.6, 809.5, 382.3, 269.8, 14.0], 'P0W': [634.6, 784.9, 376.0, 324.4, 14.8], 'K4K': [567.2, 806.2, 454.8, 303.9, 14.5], 'K7V': [567.2, 806.2, 454.8, 303.9, 14.8], 'K4R': [567.2, 806.2, 454.8, 303.9, 14.6], 'K0G': [567.2, 806.2, 454.8, 303.9, 14.1], 'N3Y': [579.8, 846.3, 415.8, 272.9, 14.9], 'K0E': [524.7, 760.7, 394.5, 279.4, 16.1], 'K2S': [567.2, 806.2, 454.8, 303.9, 14.7], 'N5A': [549.1, 772.6, 391.5, 234.4, 14.6], 'P2B': [579.5, 738.9, 336.8, 272.6, 14.7], 'P8T': [634.6, 784.9, 376.0, 324.4, 14.3], 'K0C': [567.2, 806.2, 454.8, 303.9, 18.5], 'N4Z': [549.1, 772.6, 391.5, 234.4, 14.6], 'K7A': [567.2, 806.2, 454.8, 303.9, 14.7], 'N4X': [503.1, 720.4, 361.2, 207.4, 14.6], 'L4A': [567.5, 784.9, 440.0, 276.3, 15.1], 'N7G': [503.1, 720.4, 361.2, 207.4, 14.7], 'N9K': [503.1, 720.4, 361.2, 207.4, 15.7], 'N4G': [503.1, 720.4, 361.2, 207.4, 15.1], 'L3T': [567.5, 784.9, 440.0, 276.3, 12.2], 'P0N': [583.0, 705.0, 303.0, 267.0, 14.2], 'L4J': [567.5, 784.9, 440.0, 276.3, 12.2], 'K8V': [524.7, 760.7, 394.5, 279.4, 15.4], 'L9P': [567.5, 784.9, 440.0, 276.3, 15.1], 'N8A': [503.1, 720.4, 361.2, 207.4, 15.5], 'N0B': [549.1, 772.6, 391.5, 234.4, 15.3], 'L0A': [549.1, 772.6, 391.5, 234.4, 15.3], 'L0H': [567.5, 784.9, 440.0, 276.3, 14.6], 'L9Z': [524.7, 760.7, 394.5, 279.4, 15.1], 'L0C': [567.5, 784.9, 440.0, 276.3, 14.7], 'L0V': [567.5, 784.9, 440.0, 276.3, 14.7], 'L8B': [579.8, 846.3, 415.8, 272.9, 15.4], 'N0A': [567.5, 784.9, 440.0, 276.3, 14.8], 'M9N': [567.5, 784.9, 440.0, 276.3, 14.6]}

if postal_code not in city_cf_sh: #if user entered postal code is not in system, print error and default to toronto
    print("ERROR: postal code is not in system")
    postal_code = 'M5W'

city_data = city_cf_sh.get(postal_code)
Ha = (city_data[4]/100)  # yearly capacity factor
Sh = [city_data[0], city_data[1], city_data[2], city_data[3]] # number of sun hours per season
Avg_Sh = np.mean([Sh])  # average sun hours in year 0

# calculating seasonal CF values wrt seasonal sun hours
H_0 = Ha + ((Sh[0] - Avg_Sh) * (Ha / Avg_Sh))
H_1 = Ha + ((Sh[1] - Avg_Sh) * (Ha / Avg_Sh))
H_2 = Ha + ((Sh[2] - Avg_Sh) * (Ha / Avg_Sh))
H_3 = Ha + ((Sh[3] - Avg_Sh) * (Ha / Avg_Sh))

H = [H_0, H_1, H_2, H_3]

# find max of all seasons using E @ t=0
# number of solar panels needed to fulfill at least 100% of electricity from the grid
P_0 = math.ceil((E[0][0] / L[0]) / (P * H[0] *24) * 0.35)
P_1 = math.ceil((E[0][1] / L[1]) / (P * H[1] *24) * 0.35)
P_2 = math.ceil((E[0][2] / L[2]) / (P * H[2] *24) * 0.35)
P_3 = math.ceil((E[0][3] / L[3]) / (P * H[3] *24) * 0.35)

Pn = max(P_0, P_1,P_2,P_3) 

# get mean demand and deterioration 
R_0_E = np.mean(E[t][0])
R_0_D = np.mean(d[t][0])
R_1_E = np.mean(E[t][1])
R_1_D = np.mean(d[t][1])
R_2_E = np.mean(E[t][2])
R_2_D = np.mean(d[t][2])
R_3_E = np.mean(E[t][3])
R_3_D = np.mean(d[t][3])

# initializing 

model = Model()

# initializing decision variable
y = model.add_var(name='y', var_type=INTEGER)  # number of solar panels

# initializing the objective function
model.objective = minimize(xsum((E[t][s] - ((y * P * H[s] * 24 * L[s]) * (1 - d[t][s]))) * G[t][s] + (m[t][s] * y) for s in range(S) for t in range(T)))

# adding constraints
model += (y * C) + F <= B  # budget constraint
model += y * Ap <= Armax  # area of roof constraint 
model += Pn - y >= 0 # fulfill demand constraint
# can't generate more electricity than needed each season on average over lifetime of the panels
model += (R_0_E - ((y * P * H[0] * 24 * L[0]) * (1 - R_0_D))) >= 0
model += (R_1_E - ((y * P * H[1] * 24 * L[1]) * (1 - R_1_D))) >= 0 
model += (R_2_E - ((y * P * H[2] * 24 * L[2]) * (1 - R_2_D))) >= 0 
model += (R_3_E - ((y * P * H[3] * 24 * L[3]) * (1 - R_3_D))) >= 0
model += y >= 0  # non-negativity constraint

# solving the MIP
status = model.optimize()

# printing solution
if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
    print("Z = {}".format(model.objective_value))
    print("y = {}".format(y.x))  # printing decision variable value
    numWatts = y.x * P
    numPanels = y.x
    print("optimal number of watts to install is: " + str(numWatts) + " W")
    if numPanels > 0:
        totalCost = (numPanels * C) + F
    else:
        totalCost = 0
    print("Total Capital Cost: $" + str(totalCost))
if status == OptimizationStatus.NO_SOLUTION_FOUND:
    print("no feasible solution :(")

# yearly grid energy cost w/o solar vs. yearly grid energy cost w/ solar (grouped bar chart)
# set width of bar
barWidth = 0.40

# set height of each bar
costsWithoutSolar = []
for t in range(T):
    costsWithoutSolarYearly = 0
    for s in range(S):
        costsWithoutSolarYearly = costsWithoutSolarYearly + E[t][s] * G[t][s]
    costsWithoutSolar.append(costsWithoutSolarYearly)
# print(costsWithoutSolar)

costsWithSolar = []
for t in range(T):
    costsWithSolarYearly = 0
    for s in range(S):
        costsWithSolarYearly = costsWithSolarYearly + max(0, (E[t][s] - ((numPanels * P * H[s] * 24 * L[s]) * (1 - d[t][s]))) * G[t][s])
    costsWithSolar.append(costsWithSolarYearly)
# print(costsWithSolar)


# set position of bar on X axis
r1 = np.arange(len(costsWithoutSolar))
r2 = [x + barWidth for x in r1]

# make the plot
plt.bar(r1, costsWithoutSolar, color='#e0e9ddff', width=barWidth, edgecolor='white', label='Cost without solar')
plt.bar(r2, costsWithSolar, color='#ffe599ff', width=barWidth, edgecolor='white', label='Cost with solar')

plt.xlabel('year')
plt.ylabel('$')
plt.title('Annual Spend on Grid Electricity')
plt.legend()
plt.show()


# seasonal cost w/ solar vs seasonal cost w/o solar
seasons = ['Spring', 'Summer', 'Fall', 'Winter']

# set height of each bar
springCostWithoutSolar = np.mean(E[t][0]) * np.mean(G[t][0])
summerCostWithoutSolar = np.mean(E[t][1]) * np.mean(G[t][1])
fallCostWithoutSolar = np.mean(E[t][2]) * np.mean(G[t][2])
winterCostWithoutSolar = np.mean(E[t][3]) * np.mean(G[t][3])

seasonalCostWithoutSolar = [springCostWithoutSolar, summerCostWithoutSolar, fallCostWithoutSolar, winterCostWithoutSolar]
# print(seasonalCostWithoutSolar)

springCostWithSolar = max(0, (np.mean(E[t][0]) - ((numPanels * P * H[0] * 24 * L[0]) * (1 - d[t][0]))) * np.mean(G[t][0]))
summerCostWithSolar = max(0, (np.mean(E[t][1]) - ((numPanels * P * H[1] * 24 * L[1]) * (1 - d[t][1]))) * np.mean(G[t][1]))
fallCostWithSolar = max(0, (np.mean(E[t][2]) - ((numPanels * P * H[2] * 24 * L[2]) * (1 - d[t][2]))) * np.mean(G[t][2]))
winterCostWithSolar = max(0, (np.mean(E[t][3]) - ((numPanels * P * H[3] * 24 * L[3]) * (1 - d[t][3]))) * np.mean(G[t][3]))

seasonalCostWithSolar = [springCostWithSolar, summerCostWithSolar, fallCostWithSolar, winterCostWithSolar]
# print(seasonalCostWithSolar)

labelLocations = np.arange(len(seasons))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(labelLocations - width/2, seasonalCostWithoutSolar, width, label='Seasonal Cost Without Solar', color='#e0e9ddff')
rects2 = ax.bar(labelLocations + width/2, seasonalCostWithSolar, width, label='Seasonal Cost With Solar', color='#ffe599ff')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('$')
ax.set_xlabel('Season')
ax.set_title('Avg. Seasonal Spend on Grid Electricity')
ax.set_xticks(labelLocations)
ax.set_xticklabels(seasons)
ax.legend()

fig.tight_layout()
plt.show()

# payback period (line graph)
year = []
for t in range(T):
    year.append(t)

savings = []
for t in range(T):
    savings.append(costsWithoutSolar[t] - costsWithSolar[t])
# print(np.sum(savings)) # total savings

yoySavings = [totalCost]
for t in range(1, T):
    yoySavings.append(max(0,yoySavings[t-1] - savings[t-1]))
# print(yoySavings)

plt.xlim(0, 25)
plt.ylim(0, totalCost + 1000)
plt.xlabel('year')
plt.ylabel('$')
plt.title('Payback Period')
plt.plot(year, yoySavings, color='#e0e9ddff', linewidth=2)
plt.show()


# carbon footprint tradeoff
options = ('Without Solar', 'With Solar')
y_pos = np.arange(len(options))

demandWithSolar = []
for t in range(T):
    demandWithSolarYearly = 0
    for s in range(S):
        demandWithSolarYearly = demandWithSolarYearly + max(0, (E[t][s] - ((numPanels * P * H[s] *24 * L[s]) * (1 - d[t][s])))) # will need to change to Gt
    demandWithSolar.append(demandWithSolarYearly)  

#print(np.sum(E))
#print(np.sum(demandWithSolar))
performance = [np.sum(E)*0.0003916, np.sum(demandWithSolar)*0.0003916]
plt.bar(y_pos, performance, align='center', alpha=0.5, color='#ffe599ff')
plt.xticks(y_pos, options)
plt.ylabel('CO2 kg')
plt.title('Carbon Footprint Tradeoff over 25 years')

plt.show()




