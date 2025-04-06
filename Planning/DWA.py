import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# Setup
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
possible_arcs, = ax.plot([], [], 'o-', color='red', lw=1, alpha=0.6) #all possible paths
best_arc, = ax.plot([], [], 'o-', color='green', lw=2) # Line for displaying the chosen path
line, = ax.plot([], [], color='blue', lw=1) # Line for displaying the path
point, = ax.plot([], [], 'x') # Goal

goal = np.array([75,75])
pose, current_vel = [0,0,0], [0,0]
dt = 0.1
time = []
path = [pose]
accel_max = [2, 5] # Max of linear and angular accelerations
vel_limits = [10, 5] # Max of linear and angular velocities
time_window = 2.0 # How far ahead we simulate the dynamic window
weights = np.array([1.0, 1.0, 0.3]) # Weights for tuning dyanmic window
obstacles = [np.array([20.0, 23.0, 8.0]), np.array([38.0, -60.0, 16.0]),]  # (x, y, radius)

# Create robot
radius = 0.5
robot = Circle((pose[0], pose[1]), radius, color='blue', zorder=2)
ax.add_patch(robot)
for i in range(len(obstacles)):
    ax.add_patch(Circle((obstacles[i][0], obstacles[i][1]), obstacles[i][2], color='green', alpha=0.3))
    ax.add_patch(Circle((obstacles[i][0], obstacles[i][1]), obstacles[i][2], color='green', alpha=0.3))


def motion_model(pose, u, dt):
    # standard motion model
    pose[0] += u[0] * np.cos(pose[2])*dt
    pose[1] += u[0] * np.sin(pose[2])*dt
    pose[2] += u[1]*dt
    return pose

def generate_DWA(vel):
    staticConstraints = [0.0, vel_limits[0], -vel_limits[1], vel_limits[1]]
    dynamicConstraints = [vel[0] - accel_max[0]*dt,
                          vel[0] + accel_max[0]*dt,
                          vel[1] - accel_max[1]*dt,
                          vel[1] + accel_max[1]*dt]
    # bounds for possible velocities
    dwa = np.array([max(staticConstraints[0], dynamicConstraints[0]),
                    min(staticConstraints[1], dynamicConstraints[1]),
                    max(staticConstraints[2], dynamicConstraints[2]),
                    min(staticConstraints[3], dynamicConstraints[3])])
    return dwa

def predict_course(pose, u):
    possible_path = [pose.copy()]
    sim_t = 0
    while sim_t <= time_window:
        pose = motion_model(pose.copy(), u, dt)
        possible_path.append(pose.copy())
        sim_t += dt
    return np.array(possible_path)

def calc_scoring_vals(path):
    end_pose = path[-1, :]
    angle_needed = np.arctan2(goal[1] - end_pose[1], goal[0] - end_pose[0])

    #returns positive values of difference of angles reguardless if it wraps around 360 or not
    # then makes it negative such that larger valuses are more desired, better for finding larger score
    heading_diff = - abs(np.arctan2(np.sin(angle_needed - end_pose[2]), np.cos(angle_needed - end_pose[2])))

    closest = np.inf
    for i in range(len(obstacles)):
        for j in range(len(path)):
            dist_to_obsticle_ctr = np.linalg.norm([obstacles[i][0] - path[j][0], obstacles[i][1] - path[j][1]]) - obstacles[i][2]
            if dist_to_obsticle_ctr < closest:
                closest = dist_to_obsticle_ctr
    
    return heading_diff, closest

def create_DWA_arcs(current_pose, current_vel, num_vel):
    "creates all possible trajectories within a ceratin velocity limits and at a number of intervals of num_vel"
    dw = generate_DWA(current_vel)
    dyn_windowL = np.linspace(dw[0], dw[1], num_vel)
    dyn_windowA = np.linspace(dw[2], dw[3], num_vel)
    arcs = []
    best_score = -np.inf
    for i in range(len(dyn_windowL)):
        for j in range(len(dyn_windowA)):
            arc = predict_course(current_pose, np.array([dyn_windowL[i], dyn_windowA[j]]))
            arcs.append(arc) #for plotting
            h_score, c_score = calc_scoring_vals(arc)
            # norm values to be range (-1,1)
            h_score /= np.pi 
            # use log to minimize importance of clearances that are very far away
            c_score = np.log(c_score+0.001) # avoids log zero
            # velocity score uses initial velocity and not final bc assuming constant velocity over the window bc it is our control input
            v_score = dyn_windowL[i]
            score = np.array([h_score,c_score,v_score]) @ weights
            if score > best_score:
                best_score = score
                best_u = np.array([dyn_windowL[i], dyn_windowA[j]])
                best_course = arc
    return best_course, best_u, arcs

def init():
    global goal
    goal = np.array([np.random.uniform(-90,90), np.random.uniform(-90,90)])
    possible_arcs.set_data([], [])
    best_arc.set_data([], [])
    line.set_data([], [])
    point.set_data(goal[0], goal[1])
    
    return possible_arcs, best_arc, line, point

# Animation function
def simulation(t):
    global path, time, current_vel, pose

    time.append(dt * len(time))
    
    # Arcs that start at the circle's center with defined number of velocities to choose from
    path_pred, current_vel, arcs = create_DWA_arcs(pose, current_vel, num_vel=7)
    pose = motion_model(pose, current_vel, dt)
        
    # Updating the drawing
    robot.center = (pose[0], pose[1])
    path.append(pose.copy())
    np_path = np.array(path)
    line.set_data(np_path[:,0], np_path[:,1])

#if time[-1] % 1.5 == 0:
    np_path_pred = np.array(path_pred)
    best_arc.set_data(np_path_pred[:,0], np_path_pred[:,1])

    #combining arcs for plotting
    arcs_x = []
    arcs_y = []
    #np.nan used for discontinuous lines so that all arcs can be plotted at once
    for arc in arcs:
        np_arc = np.array(arc)
        arcs_x.extend(np_arc[:, 0])
        arcs_x.append(np.nan)  # Separate lines
        arcs_y.extend(np_arc[:, 1])
        arcs_y.append(np.nan)
    possible_arcs.set_data(arcs_x, arcs_y)
    point.set_data(goal[0], goal[1])

    return robot, best_arc, possible_arcs, line, point

# Simulate
animation = FuncAnimation(fig, simulation, np.arange(0, 60, dt), init_func=init, interval=60, blit=True, repeat=True)
print(f"Number of lines on plot: {len(ax.lines)}")
for i, l in enumerate(ax.lines):
    print(f"Line {i}: {l.get_color()}, {l.get_linestyle()}, {l.get_xydata()[:2]}...")
plt.show()


