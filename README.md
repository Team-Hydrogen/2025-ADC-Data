# 2024 NASA App Development Challenge Data
This is Team Hydrogen's application for the 2025 NASA App Development Challenge. All files regarding the computation of data can be found in this repository.

## Data Summary

### Starting Data
- **Antenna Spherical Starting Antennae Coordinates** given in the Handbook in the format of Lat (deg) ,Long (deg) , Geodetic Terrain height (km).
- **HS Data Starting High School Data** detailing the position and velocity info of the Satellite, Earth, Moon, the line of sight and slant range of the antennae, and the timestamps collected.
- **Bonus Data Starting Bonus Data** showing the off-nominal path and does not contain line of sight or link budget

### Calculated Position-Based Data

#### Antenna Plot
Predicted positions of all the antennae taking in the rotational speed of the Earth, the tilt of the Earth, and the starting positions mentioned in antennaspherical.csv. Csv details the positions for all timestamps mentioned in hsdata.csv

#### Antenna Availability (Off-nominal/not)
Predicted Antenna needed for communication with Satellite based on the link budgets and line of sights calculated or given. NOTE:DSS24 is the default if all antennae are above 10000 kbps or are out of sight. All rows are given for timestamps provided in hsdata.csv/bonusdata.csv

#### Link Budget (Antenna Name) (Off-nominal/not)
Calculated or Given Link Budget for all timestamps provided in hsdata.csv/bonusdata.csv for the specified antenna and path. Line of Sight for the specified antenna is given in the format of 1 or 0, 1 being clear line of sight and 0 being no line of sight. There is no cap for link budget but 0 is provided for a value if there is no clear line of sight. NOTE: Line of sight assumes a non-cluttered Earth (no account for obstructions by surrounding buildings, natural barriers, etc.)

#### Slant Distance Offnominal
Calculated Slant Distance or Distance between Antenna and Satellite for all timestamps provided in bonusdata.csv. Slant Distance is given in km and calculated from the predicted position of the antennae and the Offnominal position path of the satellite. NOTE: The csv file is split up in equal intervals of 12981 rows, each detailing the slant distance between a particular antenna and the satellite. The order of the antennae follows "DSS24", "DSS34", "DSS54", "WPSA"

### Disclaimers
The data provided in this repository is not guaranteed to be free from error. Please double-check and note if any values seem incorrect in the files. Please contact @vigcode123 if different layouts/data/design changes are needed in an extra CSV file to further aid the incorporation of the data into the Unity scene.

### Future Updates
- Requested by @aaravdave: I am working on the speed up/down calculations for the animation.

### Credits
Team Hydrogen would like to thank the following people for their unwavering support and encouragement throughout the development of our application.
- Thank you to Data and Machine Learning Specialist Vignesha Jayakumar (@vigcode123) for creating the original write-up.
- Thank you to Lead Developer Aarin Dave (@aarindave) for editing this write-up for grammar, clarity, organization, and structure.
