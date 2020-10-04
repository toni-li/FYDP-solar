from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt
import numpy as np

# defining parameters
E0 = 2200000  # spring electricity usage (Wh) from user
E = [[E0, (E0 * 1.3), E0, E0]]  # seasonal electricity usage (Wh) with trend
G = [[0.00017952, 0.00017952, 0.00017952, 0.00017952]]  # cost of electricity from the grid at year 0($/Wh)
m = [[2.5, 2.5, 2.5, 2.5]]  # yearly maintenance cost ($/panel)
B = 20000  # budget from user
C = 315 * 2.80  # cost of each solar panel ($/panel) (12 modules of 60cell)
Ap = 18.9  # area of solar panel (ft^2) (40 * 68 inches)
Ar = 1700  # area of the roof (ft^2) from user
Armax = (Ar / 2) * 0.8
P = 315  # capacity of each solar panel (W) per hour
F = 2500  # fixed costs of installing solar panels
d = []  # deterioration factor at year i (%)
T = 25  # lifespan of solar panels
S = 4  # 0, 1, 2, 3 = “Spring”, “Summer”, “Fall”, “Winter”
L = [92, 92, 91, 90]  # number of days within each quarter

J = [[0.00010302, 0.00010302, 0.00010302,
      0.00010302]]  # cost of electricity from the grid at year 0 during off-peak hours

Pb = 13500  # battery capacity from user (W)
DoD = 0.8  # depth of discharge for battery system (%)

# filling in cost of electricity values (remain constant throughout seasons)
for t in range(1, T):
    yearly_cost = G[t - 1][0] + (G[t - 1][0] * 0.02)
    G.append([yearly_cost, yearly_cost, yearly_cost, yearly_cost])

# filling in cost of off-peak electricity values (remain constant throughout seasons)
for t in range(1, T):
    yearly_cost = J[t - 1][0] + (J[t - 1][0] * 0.2)
    J.append([yearly_cost, yearly_cost, yearly_cost, yearly_cost])

# filling in depreciation values (remain constant throughout seasons)
for t in range(T):
    yearly_depreciation = (0.0007 / 4) * t
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
for t in range(1, T):
    quarterly_maintainence = (2.5 / ((1 + i) ** t))
    m.append([quarterly_maintainence, quarterly_maintainence, quarterly_maintainence, quarterly_maintainence])

# capacity factor each season TO-DO
Ha = 0.146  # yearly capacity factor
Sh = [567.5, 784.9, 440.0, 276.3]  # number of sun hours per season
Avg_Sh = np.mean([Sh])  # average sun hours in year 0

# calculating seasonal CF values wrt seasonal sun hours
H_0 = Ha + ((Sh[0] - Avg_Sh) * (Ha / Avg_Sh))
H_1 = Ha + ((Sh[1] - Avg_Sh) * (Ha / Avg_Sh))
H_2 = Ha + ((Sh[2] - Avg_Sh) * (Ha / Avg_Sh))
H_3 = Ha + ((Sh[3] - Avg_Sh) * (Ha / Avg_Sh))

H = [H_0, H_1, H_2, H_3]
Hs = [H]
for t in range(1, T):
    Hs.append(H)

# calculating the amount of extra hours the house can use solar energy due to battery system
hourlyE = (np.mean(E[0]) / 90) / 24  # household demand in watt/hours
xstorage = Pb * (1 / hourlyE) * DoD

# find max of all seasons using E @ t=0
# number of solar panels needed to fulfill at least 100% of electricity from the grid
P_0 = math.ceil((E[0][0] / L[0]) / (P * H[0] * 24) * (0.35 + (xstorage / 24)))
P_1 = math.ceil((E[0][1] / L[1]) / (P * H[1] * 24) * (0.35 + (xstorage / 24)))
P_2 = math.ceil((E[0][2] / L[2]) / (P * H[2] * 24) * (0.35 + (xstorage / 24)))
P_3 = math.ceil((E[0][3] / L[3]) / (P * H[3] * 24) * (0.35 + (xstorage / 24)))
Pn = max(P_0, P_1, P_2, P_3)
# print("Pn:" + str(Pn))

# checking if Pn is going to generate more than the daily demand
Pnom_0 = math.ceil((E[0][0] / L[0]) / (P * H[0] * 24))
Pnom_1 = math.ceil((E[0][1] / L[1]) / (P * H[1] * 24))
Pnom_2 = math.ceil((E[0][2] / L[2]) / (P * H[2] * 24))
Pnom_3 = math.ceil((E[0][3] / L[3]) / (P * H[3] * 24))
Pnom = max(Pnom_0, Pnom_1, Pnom_2, Pnom_3)
# print("Pnom:" + str(Pnom))

if Pn > Pnom:
    Pn = Pnom


# print("Final Pn:" + str(Pn))

# setting up the model
# defining objective function
def objective(decisionVars):
    y = decisionVars[0]
    x = [[decisionVars[1], decisionVars[2], decisionVars[3], decisionVars[4]]]

    for i in range(5, T*S, 4):
            x.append([decisionVars[i], decisionVars[i+1], decisionVars[i+2], decisionVars[i+3]])
    firstTerm = 0
    secondTerm = 0
    thirdTerm = 0
    for t in range(T):
        for s in range(S):
            firstTerm = firstTerm + (
                    (0.35 * E[t][s] - ((y * P * H[s] * L[s] * 24) * (1 - d[t][s]))) * G[t][s] * x[t][s])
    for t in range(T):
        for s in range(S):
            secondTerm = secondTerm + (0.65 * E[t][s] - (((y * P * H[s] * L[s] * 24) * (1 - d[t][s])) - 0.35 * E[t][s]))
    for t in range(T):
        for s in range(S):
            thirdTerm = thirdTerm + (m[t][s] * y)

    return firstTerm + secondTerm + thirdTerm


# defining constraints
def constraint1(decisionVars):  # budget constraint
    y = decisionVars[0]
    return (-1) * y * C + F - B  # adding in the * -1 to flip the inequality to be >= 0
def constraint2(decisionVars):  # area of roof constraint
    y = decisionVars[0]
    return (-1) * y * Ap - Ar # adding in the * -1 to flip the inequality to be >= 0
def constraint3(decisionVars): # optimal number of panels should not exceed the number of panels needed to fulfill highest demand constraint
    y = decisionVars[0]
    return (-1) * y - ((0.35 * E[0][1])/(P * H[1] * L[1] * 24))
def constraint4a(decisionVars): # seasonal excess per day cannot exceed the capacity of the battery constraint for Spring
    y = decisionVars[0]
    return (-1) * ((((y * P * H[0] * L[0] * 24) * (1 - d[24][0])) - 0.35 * E[24][0])/L[0]) - (Pb * DoD)
def constraint4b(decisionVars): # seasonal excess per day cannot exceed the capacity of the battery constraint for Summer
    y = decisionVars[0]
    return (-1) * ((((y * P * H[1] * L[1] * 24) * (1 - d[24][1])) - 0.35 * E[24][1])/L[1]) - (Pb * DoD)
def constraint4c(decisionVars): # seasonal excess per day cannot exceed the capacity of the battery constraint for Fall
    y = decisionVars[0]
    return (-1) * ((((y * P * H[2] * L[2] * 24) * (1 - d[24][2])) - 0.35 * E[24][2])/L[2]) - (Pb * DoD)
def constraint4d(decisionVars): # seasonal excess per day cannot exceed the capacity of the battery constraint for Winter
    y = decisionVars[0]
    return (-1) * ((((y * P * H[3] * L[3] * 24) * (1 - d[24][3])) - 0.35 * E[24][3])/L[3]) - (Pb * DoD)
def constraint5a(decisionVars): # seasonal excess cannot exceed off-peak demand constraint for Spring
    y = decisionVars[0]
    return 0.65 * E[0][0] - (((y * P * H[0] * L[0] * 24) * (1 - d[0][0])) - (0.35 * E[0][0]))
def constraint5b(decisionVars): # seasonal excess cannot exceed off-peak demand constraint for Summer
    y = decisionVars[0]
    return 0.65 * E[0][1] - (((y * P * H[1] * L[1] * 24) * (1 - d[0][1])) - (0.35 * E[0][1]))
def constraint5c(decisionVars): # seasonal excess cannot exceed off-peak demand constraint for Fall
    y = decisionVars[0]
    return 0.65 * E[0][2] - (((y * P * H[2] * L[2] * 24) * (1 - d[0][2])) - (0.35 * E[0][2]))
def constraint5d(decisionVars): # seasonal excess cannot exceed off-peak demand constraint for Winter
    y = decisionVars[0]
    return 0.65 * E[0][3] - (((y * P * H[3] * L[3] * 24) * (1 - d[0][3])) - (0.35 * E[0][3]))


# setting constraints
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
con3 = {'type': 'ineq', 'fun': constraint3}
con4a = {'type': 'ineq', 'fun': constraint4a}
con4b = {'type': 'ineq', 'fun': constraint4b}
con4c = {'type': 'ineq', 'fun': constraint4c}
con4d = {'type': 'ineq', 'fun': constraint4d}
con5a = {'type': 'ineq', 'fun': constraint5a}
con5b = {'type': 'ineq', 'fun': constraint5b}
con5c = {'type': 'ineq', 'fun': constraint5c}
con5d = {'type': 'ineq', 'fun': constraint5d}
cons = [con1, con2, con3, con4a, con4b, con4c, con4d, con5a, con5b, con5c, con5d]

# setting bounds
bnds = [[0, None]]
for t in range(T):
    for s in range(S):
        bnds.append([0.0, 1.0])

# initial guess
initialGuess = [109]
for t in range(T*S):
    initialGuess.append(1)

sol = minimize(objective, initialGuess, method='SLSQP', bounds=bnds, constraints=cons)

# rounding up the total number of solar panels to invest in
numPanels = math.ceil(sol.x[0])

print("Z = " + str(sol.fun))
print("The optimal number of solar panels to install is: " + str(numPanels) + " panels")
print(sol.x)