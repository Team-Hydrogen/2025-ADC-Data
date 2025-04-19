import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

NOMINAL_MISSION_FILE_PATH = "Spreadsheets/Mission/Nominal/NominalMission.csv"
OFF_NOMINAL_MISSION_FILE_PATH = "Spreadsheets/Mission/OffNominal/OffNominalMission.csv"

nominal_df = pd.read_csv(NOMINAL_MISSION_FILE_PATH)
off_nominal_df = pd.read_csv(OFF_NOMINAL_MISSION_FILE_PATH)
print(off_nominal_df.head())

# Apply normalization techniques.
for column in off_nominal_df.columns[1:4]: 
    off_nominal_df[column] = (off_nominal_df[column] - off_nominal_df[column].min()) / (off_nominal_df[column].max() - off_nominal_df[column].min()) 

for column in nominal_df.columns[1:4]: 
    nominal_df[column] = (nominal_df[column] - nominal_df[column].min()) / (nominal_df[column].max() - nominal_df[column].min()) 

print(nominal_df.head())

fig, axs = plt.subplots(2, subplot_kw=dict(projection='3d'))
# ax = fig.add_subplot(projection='3d')
# ax2 = fig.add_subplot(projection='3d')
# axs[0].plot(df["Rx(km)[J2000-EARTH]"], df["Ry(km)[J2000-EARTH]"], df["Rz(km)[J2000-EARTH]"])
axs[0].plot(off_nominal_df["Rx(km)[J2000-EARTH]"], off_nominal_df["Ry(km)[J2000-EARTH]"], off_nominal_df["Rz(km)[J2000-EARTH]"])
axs[1].plot(off_nominal_df["Vx(km/s)[J2000-EARTH]"], off_nominal_df["Vy(km/s)[J2000-EARTH]"], off_nominal_df["Vz(km/s)[J2000-EARTH]"])


plt.show()



# def next_frame(iteration):
#     # print([df.iloc[iteration]["Rx(km)[J2000-EARTH]"], df.iloc[iteration]["Ry(km)[J2000-EARTH]"], df.iloc[iteration]["Rz(km)[J2000-EARTH]"]])
#     return [df.iloc[iteration]["Rx(km)[J2000-EARTH]"], df.iloc[iteration]["Ry(km)[J2000-EARTH]"], df.iloc[iteration]["Rz(km)[J2000-EARTH]"]]

# fig = plt.figure()
# ax = p3.Axes3D(fig)
# ani = animation.FuncAnimation(fig, next_frame, len(df.index),
#                                     interval=50, blit=False, repeat=True)
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=30, bitrate=1800)
# ani.save('3d-scatted-animated.mp4', writer=writer)

# plt.show()