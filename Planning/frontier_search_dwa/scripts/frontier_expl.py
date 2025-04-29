#!/usr/bin/python3

import numpy as np
from online_planning import StateValidityChecker

class frontier:
    def __init__(self, descritized_map, current_pose, vpDist_w, fSize_w):
        self.map = descritized_map.T
        self.visited = set()    #used for bfs
        self.pose = current_pose
        self.weights = np.array([vpDist_w, fSize_w])  #set weights for how to decide viewpoint (distance or number within a group of frontiers)

    def get_neighbors(self, cell, connectivity=4):
        if connectivity == 8:
            neighboring_dir = [(-1,1),  (0,1),  (1,1),
                               (-1,0),  (0,0),  (0,1),
                               (-1,-1), (0,-1), (1,-1)]
        elif connectivity == 4:
            neighboring_dir = [(0,1), (0,-1), (-1,0), (1,0)]
        else:
            print("wrong connectivity value")
            return
        neighbors = []
        for dx, dy in neighboring_dir:
            x, y = cell[0] + dx, cell[1] + dy
            if 0 <= x <= self.map.shape[0]-1 and 0 <= y <= self.map.shape[1]-1:
                neighbors.append((x,y))
        return neighbors
    
    def get_frontier_cells(self):
        frontiers = set()
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if self.map[i,j] == 0:
                    for x, y in self.get_neighbors((i,j)):
                        if self.map[x,y] == -1:
                            frontiers.add((i,j))
        print('Frontiers gathered...')
        return list(frontiers)
    
    def breadth_first_search(self, start, points):
        queue = [start]
        group = []
        while len(queue) > 0:
            point = queue.pop(0)
            if point not in self.visited:
                self.visited.add(point)
                group.append(point)
                for neighbor in self.get_neighbors(point, connectivity=8):
                    if neighbor in points and neighbor not in self.visited:
                        queue.append(neighbor)
        return group
    
    def group_frontiers(self):
        self.visited = set()
        frontiers = set(self.get_frontier_cells())
        grouped_f = []
        for frontier in frontiers:
            if frontier not in self.visited:
                group = self.breadth_first_search(frontier, frontiers)
                grouped_f.append(group)
        print(f"Frontiers grouped... {len(grouped_f)} groups found")
        return grouped_f, frontiers
    
    def choose_vp(self, travel_dist):
        grouped_f, frontiers = self.group_frontiers()

        group_dist = []
        group_size = []
        ideal_vps = []
        groups = []

        for group in grouped_f:
            group_size.append(len(group))
            closest = np.inf
            good_vp = None
            for vp in group:
                dist = np.linalg.norm(np.array(vp)-np.array(self.pose))
                if  dist < closest and dist > travel_dist:
                    closest = dist
                    good_vp = vp
                    good_group = group

            if good_vp is None:
                continue
            groups.append(good_group)
            ideal_vps.append(good_vp)
            group_dist.append(closest)
        
        highscore = -np.inf
        best_vp = None
        largest_frontier = max(group_size)
        farthest_frontier = max(group_dist)
        print('biggest frontier is:  ', largest_frontier, ' frontiers')
        print('closest frontier is:  ', min(group_dist), ' cells away')

        for i in range(len(group_dist)):
            values = np.array([1-group_dist[i]/farthest_frontier, group_size[i]/largest_frontier])
            score = np.dot(values, self.weights)
            if score > highscore:
                highscore = score
                best_vp = ideal_vps[i]
                best_group = groups[i]
        print(f'Viewpoint selected at {best_vp}')
        return best_vp, best_group, frontiers
        
test=False
if test:
    test_map = np.zeros((18,18))
    for i in range(test_map.shape[0]):
        for j in range(test_map.shape[1]):
            if i < 3 or j < 3 or i > test_map.shape[0]-4 or j > test_map.shape[1]-4:
                test_map[i,j] = -1
            else:
                test_map[i,j] = np.random.choice(np.array([1,-1,0]))


    find_frontiers = frontier(test_map, (4,4), 1.0, 0.0)
    print(test_map)
    print(find_frontiers.get_frontier_cells(), '\n')
    print(find_frontiers.choose_vp(3), '\n')
    print(np.count_nonzero(test_map == -1))
