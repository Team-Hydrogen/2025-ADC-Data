import matplotlib.pyplot as plt
import pandas as pd

NOMINAL_MISSION_FILE_PATH = "Spreadsheets/Mission/Nominal/NominalMission.csv"
OFF_NOMINAL_MISSION_FILE_PATH = "Spreadsheets/Mission/OffNominal/OffNominalMission.csv"

GEOCENTRIC_X = "Rx(km)[J2000-EARTH]"
GEOCENTRIC_Y = "Ry(km)[J2000-EARTH]"
GEOCENTRIC_Z = "Rz(km)[J2000-EARTH]"

nominal_df = pd.read_csv(NOMINAL_MISSION_FILE_PATH)
off_nominal_df = pd.read_csv(OFF_NOMINAL_MISSION_FILE_PATH)

figure = plt.figure()

ax = figure.add_subplot(111, projection='3d')

ax.plot(
    nominal_df[GEOCENTRIC_X],
    nominal_df[GEOCENTRIC_Y],
    nominal_df[GEOCENTRIC_Z],
    label='Nominal Trajectory', color='blue')
ax.plot(
    off_nominal_df[GEOCENTRIC_X],
    off_nominal_df[GEOCENTRIC_Y],
    off_nominal_df[GEOCENTRIC_Z],
    label='Off-nominal Trajectory', color='orange')

ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')

plt.legend()
plt.show()
