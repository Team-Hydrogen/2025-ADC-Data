import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Spreadsheets/Antenna/Coordinates/AntennaCartesian.csv")
df2 = pd.read_csv("Spreadsheets/Mission/Nominal/NominalMission.csv")

ANGULAR_VELOCITY = 7.2921150e-5
EARTH_TILT = np.radians(23.439292)

times_minutes = df2["MISSION ELAPSED TIME (min)"].to_list()
times = [i*60 for i in times_minutes]

antenna_coords = [[[] for i in range(3)] for _ in range(4)]
print(antenna_coords)

antenna_pos_dict = {"Antenna":[], "Time (sec)":[], "X":[], "Y":[], "Z":[]}

for i, antenna in enumerate(antenna_coords):
    X = df.iloc[i]["X"]*np.cos(EARTH_TILT) - df.iloc[i]["Z"]*np.sin(EARTH_TILT)
    Y = df.iloc[i]["Y"]
    Z = df.iloc[i]["Z"]*np.cos(EARTH_TILT) + df.iloc[i]["X"]*np.sin(EARTH_TILT)
    for t in times:
        antenna_pos_dict["Antenna"].append(df.iloc[i]["Name"])
        antenna_pos_dict["Time (sec)"].append(t)

        antenna_pos_dict["X"].append(X * np.cos(EARTH_TILT) * np.cos(ANGULAR_VELOCITY * t) - Y * np.sin(ANGULAR_VELOCITY * t) * np.cos(EARTH_TILT) + Z * np.sin(EARTH_TILT))
        antenna_pos_dict["Y"].append(X * np.sin(ANGULAR_VELOCITY * t) + Y * np.cos(ANGULAR_VELOCITY * t))
        antenna_pos_dict["Z"].append(-X * np.sin(EARTH_TILT) * np.cos(ANGULAR_VELOCITY * t) + Y * np.sin(EARTH_TILT) * np.sin(ANGULAR_VELOCITY * t) + Z * np.cos(EARTH_TILT))
# print(antenna_coords)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for i in range(4):
    ax.plot(df.iloc[i]["X"],df.iloc[i]["Y"], df.iloc[i]["Z"],'ro',label=df.iloc[i]["Name"])
for i in range(4):
    ax.plot(antenna_pos_dict["X"][12981*i:12981*(i+1)], antenna_pos_dict["Y"][12981*i:12981*(i+1)], antenna_pos_dict["Z"][12981*i:12981*(i+1)], label=df.iloc[i]["Name"])

plt.legend()
plt.show()

df_out = pd.DataFrame(antenna_pos_dict, columns=list(antenna_pos_dict.keys()))
df_out.to_csv("Spreadsheets/Computation/AntennaPlot.csv",index=False)
