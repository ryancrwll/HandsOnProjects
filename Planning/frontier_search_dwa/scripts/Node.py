import math
import numpy as np

 
class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def dist(self, p):
        return math.sqrt((self.x - p.x) ** 2 + (self.y - p.y) ** 2)

    def numpy(self):
        return np.array([self.x, self.y])

    def dist_line(self, l):
        return np.linalg.norm(
            np.cross(l.p2.numpy() - l.p1.numpy(), l.p1.numpy() - self.numpy())
        ) / np.linalg.norm(l.p2.numpy() - l.p1.numpy())

    def to_str(self):
        return "({}, {})".format(np.round(self.x, 2), np.round(self.y, 2))

    def dot(self, p):
        return self.x * p.x + self.y * p.y

    def norm(self):
        return math.sqrt(self.x**2 + self.y**2)

    def vector(self, p):
        return Point(p.x - self.x, p.y - self.y)

    def unit(self):
        mag = self.length()
        if mag > 0:
            return Point(self.x / mag, self.y / mag)
        else:
            return Point(0, 0)

    def scale(self, sc):
        return Point(self.x * sc, self.y * sc)

    def __add__(self, p):
        return Point(self.x + p.x, self.y + p.y)

    def __sub__(self, p):
        return Point(self.x - p.x, self.y - p.y)

    def __truediv__(self, s):
        return Point(self.x / s, self.y / s)

    def __floordiv__(self, s):
        return Point(int(self.x / s), int(self.y / s))

    def __mul__(self, s):
        return Point(self.x * s, self.y * s)

    def __rmul__(self, s):
        return self.__mul__(s)

    def dist_segment(self, s):
        line_vec = s.p1.vector(s.p2)
        pnt_vec = s.p1.vector(self)
        line_len = line_vec.length()
        line_unitvec = line_vec.unit()
        pnt_vec_scaled = pnt_vec.scale(1.0 / line_len)
        t = line_unitvec.dot(pnt_vec_scaled)
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        nearest = line_vec.scale(t)
        dist = nearest.dist(pnt_vec)
        nearest = nearest.add(s.p1)
        return dist

class Node(Point):
    def __init__(self, x,  y, parent=None, dist_prev=0):
        super(Node, self).__init__(x, y)

        self.parent = parent
        self.coord = (x, y)
        self.bottom_left_coord = (x-4, y-4)
        if self.parent is not None:
            self.g = self.parent.g + dist_prev
        else: 
            self.g = 0

    def __eq__(self, node):
        return self.x == node.x and self.y == node.y
    
    def __lt__(self, other:object) -> bool:
        if not isinstance(other, Node):
            raise NotImplementedError
        return self.g < other.g
    
    def __str__(self):
        if self.parent is None:
            return "({}, {}, p: None)".format(np.round(self.x, 2), np.round(self.y, 2))
        else:
            return "({}, {}, p: {})".format(np.round(self.x, 2), np.round(self.y, 2), (self.parent.x,self.parent.y))

    def __repr__(self):
        if self.parent is None:
            return "({}, {}, p: None)".format(np.round(self.x, 2), np.round(self.y, 2))
        else:
            return "({}, {}, p: {})".format(np.round(self.x, 2), np.round(self.y, 2), (self.parent.x,self.parent.y))
        
    

    
