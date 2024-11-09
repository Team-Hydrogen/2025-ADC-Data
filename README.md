# 2024 NASA App Development Challenge Data
All files relating to the computation of the data provided

## Data Summary

### Starting Data
1. Antenna Spherical
Starting Antennae Coordinates given in the Handbook in the format of Lat (deg) ,Long (deg) , Geodetic Terrain height (km).
2. HS Data
Starting High School Data detailing the position and velocity info of the Satellite, Earth, Moon, the line of sight and slant range of the antennae, and the timestamps collected.
3. Bonus Data
Starting Bonus Data showing the off-nominal path and does not contain line of sight or link budget

### Calculated Position-Based Data
1. Antenna Plot
Predicted positions of all the antennae taking in the rotaional speed of the Earth, the tilt of the Earth, and the starting positions mentioned in antennaspherical.csv. Csv details the positions for all timestamps mentioned in hsdata.csv
2. Antenna Availability (Offnominal/not):
Predicted Antenna needed for communication with Satellite based on the link budgets and line of sights calculated or given. NOTE:DSS24 is the default if all antennae are above 10000 kbps or are out of sight. All rows are given for timestamps provided in hsdata.csv/bonusdata.csv
3. Link Budget (Antenna Name) (Offnominal/not):
Calculated or Given Link Budget for all timestamps provided in hsdata.csv/bonusdata.csv for the specified antenna and path. Line of Sight for the specified antenna is given in the format of 1 or 0, 1 being clear line of sight and 0 being no line of sight. There is no cap for link budget but 0 is provided for a value if there is no clear line of sight. NOTE: Line of sight assumes a non-cluttered Earth (no account for obstructions by surrounding buildings, natural barriers, etc.)
4. Slant Distance Offnominal:
Calculated Slant Distance or Distance between Antenna and Satellite for all timestamps provided in bonusdata.csv. Slant Distance is given in km and calculated from the predicted position of the antennae and the Offnominal position path of the satellite. NOTE: The csv file is split up in equal intervals of 12981 rows, each detailing the slant distance between a particular antenna and the satellite. The order of the antennae follows "DSS24", "DSS34", "DSS54", "WPSA"

Please double-check and note if any values seem incorrect in the files. Please contact if different layouts/data/design changes is needed in an extra csv to further aid the incorporation of the data into the Unity scene. Thank you!

PS: Yes Aarav, I am working on the speed up/down calculations for the animation.

Vignesha Jayakumar
