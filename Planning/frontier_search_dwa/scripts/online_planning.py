#!/usr/bin/python3

import string
from turtle import distance
import numpy as np
import math as m

from Node import Node
from scipy.spatial import KDTree
import copy
import random 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap


def wrap_angle(angle): 
    return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )

def dist_between_points (p1, p2):
    if type(p1) == Node:
        return m.sqrt((p2.coord[0] - p1.coord[0])**2 + (p2.coord[1] - p1.coord[1])**2)
    else:
        return m.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

class StateValidityChecker:
    """ Checks if a position or a path is valid given an occupancy map."""

    # Constructor
    def __init__(self, distance, is_unknown_valid=True):
        # map: 2D array of integers which categorizes world occupancy
        self.map = None 
        # map sampling resolution (size of a cell))                            
        self.resolution = None
        # world position of cell (0, 0) in self.map                      
        self.origin = None
        # set method has been called                          
        self.there_is_map = False
        # radius arround the robot used to check occupancy of a given position                 
        self.distance = distance                    
        # if True, unknown space is considered valid
        self.is_unknown_valid = is_unknown_valid    

    def map_viz_debug(self, env, title: string, pos=np.array([None]), goal=np.array(None)):
        cmap = ListedColormap(['white', 'black', 'gray'])
        map_viz = np.zeros_like(env, dtype=int)
        map_viz[env < 0] = 2
        map_viz[env == 0] = 1
        map_viz[env > 0] = 0

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(map_viz, cmap=cmap, origin='lower')
        if pos[0] != None:
            distance_pixel = int(self.distance / self.resolution)+1
        
            fromx, fromy = pos[0]-distance_pixel, pos[1]-distance_pixel
            tox, toy = pos[0] + distance_pixel, pos[1]+distance_pixel
            rect = patches.Rectangle((fromx, fromy), tox-fromx, toy-fromy, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        if goal[0] != None:
            distance_pixel = int(self.distance / self.resolution)+1
        
            fromx, fromy = goal[0]-distance_pixel, goal[1]-distance_pixel
            tox, toy = goal[0] + distance_pixel, goal[1]+distance_pixel
            rect = patches.Rectangle((fromx, fromy), tox-fromx, toy-fromy, linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(False)
        plt.show()

    def map_set(self, map):
        bin_map = np.zeros_like(map)
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                if map[i,j] == 100:
                    bin_map[i,j] = 1
                if map[i,j] == -1:
                    bin_map[i,j] = -1
        # self.map_viz_debug(bin_map, 'before rotate and flip')
        bin_map = self.rotate_and_flip(bin_map)
        # self.map_viz_debug(bin_map, 'map_set')
        return np.array(bin_map)
    
    # Set occupancy map, its resolution and origin. 
    def set(self, data, resolution, origin):
        self.map = self.map_set(data)
        #self.map_viz_debug(self.map, 'set', self.current_pose[0:2])
        self.resolution = resolution
        self.origin = np.array(origin)
        self.there_is_map = True

    # Transform position with respect the map origin to cell coordinates
    def position_to_map(self, point):
        origin = self.origin
        resolution = self.resolution
        shape = self.map.shape  # map shape is height, width

        # this function variates from the requiered, since it asumes that the reference from the grid map is in the top left corner 
        x,y = point-origin #x,y are already in map (occupancy grid) coordinates
        xmap, ymap = shape[1] - int(x/resolution), shape[0] - int(y/resolution)
        if x < 0 or y < 0 or x > shape[1] or y > shape[0]: 
            print ("error while converting to gridmap")
            return None
        else: 
            return np.array([xmap,ymap])

    def map_to_position(self, m):
        origin = self.origin
        resolution = self.resolution
        shape = self.map.shape
        if m[0] < 0 or m[1] < 0 or m[0] >= shape[1] or m[1] >= shape[0]: 
            print("error while converting to position")
            return None
        
        x,y = (shape[1]-m[0])*resolution, (shape[0]-m[1])*resolution 
        x,y = x + origin[0],y+ origin[1]
        return np.array((x,y))
    

    def rotate_and_flip(self, matrix):
        # Rotate the matrix 90 clockwise (Transpose and reverse each row)
        matrix = np.array(matrix).T[::-1]
        
        # Flip the matrix horizontally (reverse the columns)
        matrix = matrix[:, ::-1]

        return matrix
    
    def collision_free(self, point1, point2, plot=False):
        num_steps = max(abs(point2[0] - point1[0]), abs(point2[1] - point1[1])) +1
        # Interpolate x and y coordinates
        x_values = np.linspace(point1[0], point2[0], num_steps+10)
        y_values = np.linspace(point1[1], point2[1], num_steps+10)
        
        # Combine x and y into pixel coordinates
        pixels = list(zip((np.floor(x_values)).astype(int), (np.floor(y_values)).astype(int)))
        if len(pixels) == 0: return False
        for i in range(len(pixels)):

            reply = self.is_valid_pixel(pixels[i])

            if reply == False: return False
        return True
        
    def is_valid_pixel (self, pixel, added_pixel_assurance = 1, free_space_goal = False, plot = False):

        # y is row, x is column
        if free_space_goal:
            second_check = -1 # invalidates unknown spaces
        else:
            second_check = np.inf # value the map will never be
        distance_pixel = int(self.distance / self.resolution)+added_pixel_assurance
        
        fromx, fromy = pixel[0]-distance_pixel, pixel[1]-distance_pixel
        tox, toy = pixel[0] + distance_pixel, pixel[1]+distance_pixel # get x,y pixels from bottom right corner of the robot threshold
        
        if int(m.floor(fromx)) < 0 or int(m.floor(fromy)) < 0: return False
        if int(round(tox)) > self.map.shape[1] or int(m.ceil(toy)) > self.map.shape[0]: return False
        if plot:
            fig, ax = plt.subplots()
            ax.imshow(self.map, cmap='gray', origin='upper')
            rect = patches.Rectangle((fromx, fromy), tox-fromx, toy-fromy, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.show()

            print (self.map[fromy:toy,fromx:tox])

        for i in range(int(round(fromy)), int(round(toy))):
            for j in range(int(round(fromx)+1), int(round(tox))):
                if self.map[i,j] == 1 or self.map[i,j] == second_check: 
                    return False 
        return True
    
    def check_path(self, path):
        if len(path) < 2:
            #path is just two locations already checked by RRT
            return True
        for i in range(len(path)-1):
            if not self.collision_free(self.position_to_map(path[i]), self.position_to_map(path[i+1])):
                print ("Collision between ", path[i], path[i+1])
                return False
        return True
    


        
# Define Planner class (you can take code from Autonopmous Systems course!)
class Planner:
    def  __init__(self, state_validity_checker, max_iterations, dominion=[-10, 10, -10, 10], goal=None, start=None, goal_threshold=0.05, smooth=True):
        # define constructor ...
        self.max_iterations = max_iterations
        self.goal = Node(goal[0], goal[1])##
        self.start = Node(start[0], start[1])
        self.tree = [self.start]
        self.robot_distance = state_validity_checker.distance
        self.map_resolution = state_validity_checker.resolution
        self.goal_threshold = goal_threshold
        self.svc = state_validity_checker
        self.step_size = 3
        self.map = state_validity_checker.map
        self.smooth = smooth
        pass
    
    def compute_path(self):
        if  self.svc.is_valid_pixel(self.start.coord, plot=False) == False:
            raise AssertionError("Start is in collision, Please restart the simulation")
            pass
        #Goal collision check done in plan()
        for k in range(self.max_iterations):
            # 1. Sample a random node
            random_node = self.get_random_node(0.2)

            # 2. Find the nearest node in the tree
            nearest, q_dist, index = self.nearest_node(random_node)
            
            # 3. New configuration
            qnew, q_dist = self.new_config(nearest, random_node, q_dist, self.step_size)
            # 4. Check if the new configuration is collision free
            if self.svc.collision_free((nearest.x,nearest.y), (qnew.x,qnew.y)):

                qnew = self.new_rrtStar_node(qnew, nearest, radius = self.step_size*2)
                qnew.parent = nearest

                # 5. Check if goal is reached
                self.tree.append(qnew)
                
                if dist_between_points(qnew, self.goal) <= self.goal_threshold:
                    path1, node_path = self.build_path(qnew)
                    if self.smooth:
                        path = self.smooth_path(node_path)
                        return self.tree, path  # The robot is already at the first node, so we skip it
                    return self.tree, path1
                
        x_values = [node.x for node in self.tree]
        y_values = [node.y for node in self.tree]
        """ plt.imshow(self.map, cmap='gray', origin='upper')  # Display the binary map
        plt.colorbar(label="Binary Map Value")  # Optional: color bar for reference
        plt.plot(x_values, y_values, marker='o', color='g', linestyle="" ,markersize=3, label="Tree")  # Path as a red line with circles
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Binary Map with Path')
        plt.legend()
        plt.show() 
        plt.grid() """

        raise AssertionError("Max iterations reached. No path found")
        

    def smooth_path(self, nodes):
        nodes = nodes[::-1]
        smooth_path = []
        target = nodes[-1]        
        i = 0
        path = []
        while target is not nodes[0]:
            node = nodes[i]
            i += 1
            if self.svc.collision_free((node.x, node.y), (target.x, target.y)):
                i = 0
                smooth_path.append(target)
                path.append(target.coord)
                smooth_path[-1].parent = node
                target = node  
        smooth_path.append(nodes[0])
        path.append(nodes[0].coord)  


        return path

    def get_random_node(self,p):
        if random.random() < p:
            return self.goal
        # Get all free or unknown cells (where grid is 0 or -1)
        free_positions = np.argwhere(self.map <= 0)
        # Randomly choose one of the free cells 
        random_index = np.random.choice(len(free_positions))
        node = Node(free_positions[random_index][0],free_positions[random_index][1])

        return node
    
    # def nearest_node(self, random_node):
    #     q_near, index = min(self.tree, key=lambda node: dist_between_points(node, random_node))
    #     return q_near, index
    
    def nearest_node(self, q_rand: Node):
        # Find the nearest neighbor to q_rand using KD Trees
        nodelist =  [node.coord for node in self.tree]
        tree = KDTree(nodelist)
        dist, index = tree.query(q_rand.coord, k=1)
        q_near = self.tree[index]

        return q_near, dist, index    
    
    def new_rrtStar_node(self, q_new: Node, node_near: Node, radius):
        within_reach = []
        indexes = []
        distances = []

        for i in range(len(self.tree)):
            dist = np.linalg.norm(np.array(self.tree[i].coord) - np.array(q_new.coord))
            distances.append(dist)
            if distances[i] <= radius:
                within_reach.append(self.tree[i])
                indexes.append(i)

        if len(within_reach) != 0:
            best_index = np.argmin(within_reach)
            node_index = indexes[best_index]
            least_cost = within_reach.pop(best_index)
            node = Node(q_new.coord[0], q_new.coord[1], dist_prev=least_cost.g, parent=self.tree[node_index])
        else:
            node = Node(q_new.coord[0], q_new.coord[1], dist_prev=node_near.g, parent=node_near)
        
        return node
    
    def new_config(self, nearest, random_node, q_dist, step_size):

        if q_dist <= step_size:
            return random_node, q_dist
            
        else:
            theta = m.atan2(random_node.coord[1] - nearest.coord[1], random_node.coord[0] - nearest.coord[0])
            return Node(int(nearest.coord[0] + step_size * m.cos(theta)), int(nearest.coord[1] + step_size * m.sin(theta))), step_size
    
    def get_edges(self,nodes):
        edges = []
        for node in nodes:
            if node.parent:
                edges.append((nodes.index(node.parent), nodes.index(node)))
        return edges
    

    def build_path(self, node):
        path = []
        node_path = []
        while node.parent is not None:
            path.append(node.coord)
            node_path.append(node)
            node = node.parent
        path.append(node.coord)
        node_path.append(node)
        return path[::-1], node_path[::-1]

# Planner: This function has to plan a path from start_p to goal_p. To check if a position is valid the 
# StateValidityChecker class has to be used. The planning dominion must be specified as well as the maximum planning time.
# The planner returns a path that is a list of poses ([x, y]).
def compute_path_global(start_p, goal_p, state_validity_checker, dominion, max_iterations= 1000):


    # TODO: Plan a path from start_p to goal_p inside dominiont using a Planner Object and the 
    # StateValidityChecker Object previously defined.
    # TODO: if solved, return a list with the [x, y] points in the solution path.
    # example: [[x1, y1], [x2, y2], ...]
    # TODO: Ensure that the path brings the robot to the goal (with a small tolerance)!
    start_p = state_validity_checker.position_to_map(start_p)
    goal_p = state_validity_checker.position_to_map(goal_p)
    planner = Planner(state_validity_checker, max_iterations, dominion, goal_p, start_p, smooth=True)
    tree, path = planner.compute_path()

    for i in range(len(path)):
        new = state_validity_checker.map_to_position(path[i])[:]
        path[i] = [new[0], new[1]]
    
    return path[1:] #remove first location because the robot is already at first position

# Controller: Given the current position and the goal position, this function computes the desired 
# lineal velocity and angular velocity to be applied in order to reah the goal.
def move_to_point(current, current_yaw, goal, Kv=0.5, Kw=0.5):
    
    # Use a proportional controller which sets a velocity command to move from current position to goal (u = Ke)
    # To avoid strange curves, first correct the orientation and then the distance. 
    # Hint: use wrap_angle function to maintain yaw in [-pi, pi]
    # This function should return only  linear velocity (v) and angular velocity (w)

    angle_error = current_yaw - m.atan2( goal[1]- current[1],  goal[0]-current[0]) 
    angle_error = wrap_angle(angle_error)
    if abs(angle_error) > 0.04 :# anglular error is greater than anglular threshold 0.0873 rad = no idea deg app 5/2
        w = angle_error * Kw
        v = 0
    else:
        w = angle_error * Kw
        v = Kv * dist_between_points(current,goal)
    
    return  v, w
