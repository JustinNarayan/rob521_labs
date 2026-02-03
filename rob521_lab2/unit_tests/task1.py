# Stop pygame warning
import warnings
warnings.simplefilter("ignore", category=UserWarning)

# Imports
import numpy as np
from rob521_lab2.nodes.l2_planning import PathPlanner

### Set up Path Planner
# map info
map_filename = "willowgarageworld_05res.png"
map_setings_filename = "willowgarageworld_05res.yaml"
# robot information
goal_point = np.array([[10], [10]])  # m
stopping_dist = 0.5  # m
# create planner
path_planner = PathPlanner(
    map_filename, map_setings_filename, goal_point, stopping_dist
)


### Point to Cell

# Cell outputs
cells = path_planner.point_to_cell(np.array([
    [-1,   0,   1],
    [-1,   0,   1]
]))
'''
The robot's reference frame is 21m right and 49.25m and up from the bottom left.
The occupancy map's resolution is 20px per m.
The occupancy map also counts down the map rather than up, and the map is 1600px tall.
The robot's y position in px is thus 1600 - (robot's y position in m converted to map res)

So @ (0m,0m) to the robot -> (21m * 20px/m, 49.25m * 20px/m) to the map

[-1, 0, 1] --> [400, 420, 440]
[-1, 0, 1]     [635, 615, 595]
                      ^
  49.25 * 20 = 985 px in map frame -> 1600 - 985 = 615 px in occupancy map frame
'''
print(cells)

# Empty
cells = path_planner.point_to_cell(np.array([
    [],
    []
]))
'''
Should be empty.
'''
print(cells)


### Points to Robot Circle

# A single cell
cells = path_planner.points_to_robot_circle(np.array([ 
    [0],
    [0]
]))
'''
For each point (there is 1), there will be a 2xM array within.
M is the number of occupid cells. Each cell's coordinates are listed.
The total shape should be 1x2xM
'''
print(np.array(cells).shape)
print(cells)

# Many cells

cells = path_planner.points_to_robot_circle(np.array([ 
    [0, -4.5,   3.333],
    [0,  1.0, -11.874]
]))
'''
Should be 3x2xM
'''
print(np.array(cells).shape)