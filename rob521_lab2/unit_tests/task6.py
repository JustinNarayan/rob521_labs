# Imports
import numpy as np
from rob521_lab2.nodes.l2_planning import PathPlanner, Node

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

"""
===========================================================
Unit Tests: update_children()
===========================================================

We will manually construct a simple tree:

        (0)
         |
        (1)
         |
        (2)
       /   \
     (3)   (4)

Then we will change node (1)'s cost and verify that all
descendants update correctly.

Edge costs are approximated in update_children() as Euclidean
distance in (x,y).

So expected costs should propagate as:

cost(child) = cost(parent) + distance(parent, child)
"""

print("\n==============================")
print("TEST 1: Build Manual Tree")
print("==============================")

# Reset nodes dictionary manually
path_planner.nodes = {}

# Create nodes manually
path_planner.nodes[0] = Node(np.array([0, 0, 0]), -1, 0)

path_planner.nodes[1] = Node(np.array([1, 0, 0]), 0, 1)
path_planner.nodes[2] = Node(np.array([2, 0, 0]), 1, 2)

path_planner.nodes[3] = Node(np.array([2, 1, 0]), 2, 3)
path_planner.nodes[4] = Node(np.array([3, 0, 0]), 2, 3)

# Manually assign children lists
path_planner.nodes[0].children_ids = [1]
path_planner.nodes[1].children_ids = [2]
path_planner.nodes[2].children_ids = [3, 4]
path_planner.nodes[3].children_ids = []
path_planner.nodes[4].children_ids = []

print("Tree constructed.")
print("Initial costs:")
for nid in path_planner.nodes:
    print(f"Node {nid} cost = {path_planner.nodes[nid].get_cost()}")

"""
At this point costs are arbitrary.
Now we will force a cost change at node (1)
and test propagation downward.
"""

print("\n==============================")
print("TEST 2: Update Children Costs")
print("==============================")

# Force node (1) cost to something different
path_planner.nodes[1].set_cost(10)

print("Changed node (1) cost to 10.")
print("Calling update_children(1)...")

# Run update_children from node 1 downward
path_planner.update_children(1)

print("\nUpdated costs after propagation:")

for nid in [1, 2, 3, 4]:
    print(f"Node {nid} cost = {path_planner.nodes[nid].get_cost()}")

"""
Expected behavior:

Node (2):
 parent = (1)
 distance = ||(2,0) - (1,0)|| = 1
 cost(2) = 10 + 1 = 11

Node (3):
 parent = (2)
 distance = ||(2,1) - (2,0)|| = 1
 cost(3) = 11 + 1 = 12

Node (4):
 parent = (2)
 distance = ||(3,0) - (2,0)|| = 1
 cost(4) = 11 + 1 = 12
"""

print("\nExpected:")
print("Node 2 cost should be 11")
print("Node 3 cost should be 12")
print("Node 4 cost should be 12")

print("\n==============================")
print("TEST 3: Deeper Propagation Check")
print("==============================")

"""
Now we extend the tree one more level:

        (3)
         |
        (5)

Then update from node (2) and verify node (5) updates too.
"""

# Add node (5) as child of node (3)
path_planner.nodes[5] = Node(np.array([2, 2, 0]), 3, 999)

# Update children list
path_planner.nodes[3].children_ids.append(5)

print("Added node (5) under node (3).")
print("Node (5) initial cost was set to 999 (wrong).")

# Now update downward from node (2)
print("Calling update_children(2)...")
path_planner.update_children(2)

print("\nUpdated costs after deeper propagation:")

print(f"Node 3 cost = {path_planner.nodes[3].get_cost()}")
print(f"Node 5 cost = {path_planner.nodes[5].get_cost()}")

"""
Expected:

Node (5):
 parent = (3)
 distance = ||(2,2) - (2,1)|| = 1
 cost(5) = cost(3) + 1

If cost(3) was 12, then:

cost(5) = 13
"""

print("\nExpected:")
print("Node 5 cost should be 13")

print("\n==============================")
print("ALL update_children TESTS DONE")
print("==============================\n")
