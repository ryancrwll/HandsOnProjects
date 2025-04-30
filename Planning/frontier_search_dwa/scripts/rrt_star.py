import numpy as np
from scipy.spatial import KDTree

### Define the custom Node class

class Node: 
    def __init__(self, p, dist_prev=0, prev=None) -> None:
        self.p = p
        self.prev = prev

        # Calculating the tentative cost of reaching the node
        if self.prev is not None:
            self.g = self.prev.g + dist_prev
        else: 
            self.g = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            raise NotImplementedError

        return np.array_equal(self.p, other.p) 
    
    def __lt__(self, other:object) -> bool:
        if not isinstance(other, Node):
            raise NotImplementedError
        
        return self.g < other.g
    
    def __hash__(self):
        return hash((self.p[0], self.p[1]))
    

### Define the RRT class

class RRT:
    def __init__(self, max_iters=10000, goal_prob=0.2, max_dist=10.0, step_size=0.75, eps=0.01):
        self.max_iters = max_iters
        self.goal_prob = goal_prob
        self.max_dist = max_dist
        self.step_size = step_size
        self.eps = eps

    def rand_conf(self, grid):
        # Sample a random number from [0, 1)
        r = np.random.rand()

        # If r < goal_prob then choose the goal as q_rand
        if r < self.goal_prob:
            q_rand = grid.get_goal()
        else:
            is_free = False
            while not is_free:
                q_rand = np.random.uniform(0, grid.max_limits, size=2)
                is_free = grid.is_free(q_rand)
        
        return q_rand
    
    def nearest_vertex(self, q_rand, q_list):
        # Find the nearest neighbor to q_rand using KD Trees
        tree = KDTree(q_list)
        dist, index = tree.query(q_rand, k=1)
        q_near = q_list[index]

        return q_near, dist, index

    def new_conf(self, q_near, q_rand, dist_near):
        # Limit the max distance to max_dist
        dist_ratio = self.max_dist / (dist_near + 1e-6)

        if dist_ratio < 1.0:
            q_new = q_near + dist_ratio * (q_rand - q_near)
            dist_near = self.max_dist
        else:
            q_new = q_rand

        return q_new, dist_near
    
    def is_segment_free(self, q_near, q_new, grid: GridMap):
        # Sample equidistant points from q_near to q_new
        distance = np.hypot(*(q_new - q_near))
        num_points = np.ceil(distance / self.step_size).astype(int)
        segment = np.linspace(q_near, q_new, num=num_points)

        return grid.is_free(segment)
    
    def rrt_algorithm(self, grid: GridMap, star=False):
        # Get start and goal coordinates and initialize start node
        start, q_goal = grid.get_start(), grid.get_goal()
        start = Node(start)

        # Initialize lists for storing nodes, vertices, and edges
        node_list = [start]
        vertex_list = [start.p]
        edge_list = []

        # Begin RRT loop
        for k in range(self.max_iters):
            q_rand = self.rand_conf(grid)
            q_near, dist_near, index_near = self.nearest_vertex(q_rand, vertex_list)
            q_new, dist_near = self.new_conf(q_near, q_rand, dist_near)
            if self.is_segment_free(q_near, q_new, grid):
                # checking if we want rrt star
                if star == True:
                # rrt star is implemented through injection of following func
                    node = self.new_rrtStar_node(q_new=q_new, node_near=node_list[index_near],
                                                 node_list=node_list, radius=self.step_size*2)
                else:
                    node = Node(q_new, dist_prev=dist_near, prev=node_list[index_near])
                node_list.append(node)
                vertex_list.append(node.p)
                edge_list.append([index_near, len(vertex_list)-1])

                if np.hypot(*(q_new - q_goal)) < self.eps:
                    path_cost = node.g 
                    path = self.compute_path(node)
                    path_length = len(path)
                    path_info = (path, path_length, path_cost)
                    print(f'Path found in {k+1} iterations \n')

                    return path_info, np.array(vertex_list), np.array(edge_list)
        
        print('No solution found :(')

    def compute_path(self, goal: Node):
        path = [goal.p]
        node = goal.prev
        while node is not None:
            path.append(node.p)
            node = node.prev
        
        return np.flip(np.array(path), axis=0)
    
    def smoothing(self, path: np.array, grid: GridMap):
        #init of new lists
        skipped_pos = []
        smoothed_path = []

        #init of indexes
        i_last = len(path)-1
        i_first = 0

        while i_last != 0:
            #checking if vertices can be skipped
            while self.is_segment_free(path[i_last], path[i_first], grid) == False:
                i_first += 1
            #adding vertices for new path and skipped nodes
            smoothed_path.append(path[i_last])
            removed = path[i_first+1:i_last]
            skipped_pos.append(removed)
            i_last = i_first
            i_first = 0
        smoothed_path.append(path[0])
        smoothed_path.reverse()
        path_length = len(smoothed_path)
        smoothed_path = np.array(smoothed_path)
        #recomputing the path cost
        cost = 0
        for i in range(1,path_length):
            cost += np.hypot(*(smoothed_path[i]-smoothed_path[i-1]))
        #calculation of number of skipped nodes (irregular size)
        len_skippedpos = sum(arr_i.size for arr_i in skipped_pos)
        smoothpath_info = (smoothed_path, path_length, cost)
        print(f' \n Number of skipped vertices: {len_skippedpos} \n')

        return smoothpath_info
    
    def new_rrtStar_node(self, q_new, node_near: Node, node_list, radius):
        within_reach = []
        indexes = []
        distances = []

        for i in range(len(node_list)):
            distances.append(np.hypot(*(node_list[i].p-q_new)))
            if distances[i] <= radius:
                within_reach.append(node_list[i])
                indexes.append(i)

        if len(within_reach) != 0:
            best_index = np.argmin(within_reach)
            node_index = indexes[best_index]
            least_cost = within_reach.pop(best_index)
            node = Node(q_new, dist_prev=least_cost.g, prev=node_list[node_index])
        else:
            node = Node(q_new, dist_prev=node_near.g, prev=node_near)
        
        return node

            

