from matplotlib import pyplot as plt
import numpy as np


def plot (data, width, height):
    data2 = [100 if x == -1 else 255 if x == 100 else x for x in data]


    map = np.array(data2).reshape(width, height)

    plt.imshow(map, cmap='bone') 
    plt.show()
    return None

def __position_to_map__(p, origin, resolution, shape):
    # this function variates from the requiered, since it asumes that the reference from the grid map is in the top left corner 
    x,y = p-origin #x,y are already in map (occupancy grid) coordinates
    xmap, ymap =  int(x/resolution), shape[1] - int(y/resolution)
    if x < 0 or y < 0 or x > shape[0] or y > shape[1]: return None
    else: return np.array([xmap,ymap])

def __map_to_position__(m, origin, resolution, shape):
    if m[0] < 0 or m[1] < 0 or m[0] > shape[0] or m[1] > shape[1]: return None
    x,y = m[0]*resolution, (-m[1] + shape[1] )*resolution 
    x,y = x + origin[0],y+ origin[1]
    return np.array((x,y))

def is_valid(p, distance, map, resolution, orig, shape):
    print ("the robot is at ", p)


    # robot area in world coordinates
    pfrom = np.array((p[0] - distance, p[1] + distance)) 
    pto = np.array((p[0] + distance, p[1] - distance))
    print ("top left corner: ", pfrom)
    print ("bottom right corner: ", pto)

    fromx, fromy = __position_to_map__(pfrom,orig,resolution,shape) # get x,y pixels from top left corner of the robot threshold
    tox, toy = __position_to_map__(pto,orig, resolution, shape) # get x,y pixels from bottom right corner of the robot threshold

    print ("top left corner pixels:", fromx,fromy)
    print ("botom right corner: ", tox,toy)
    print (map[fromx:tox,fromy:toy])
    for i in range(fromx, tox):
        for j in range(fromy, toy):
            if map[i,j] > 50: return False
   
    return True

def dist_between_points (p1, p2):
    return np.linalg.norm(p1-p2)
