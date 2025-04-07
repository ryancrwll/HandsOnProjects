matplotlib simulation of functioning dwa algorithm
utilizes functions from slides with added scoring and weight for distance to goal to reduce circling the goal
key functions:
    motion_model(): uses unicycle model need to update for diff drive model
    generate_DWA(): creates max and minimum values for linear and angular velocities, uses physics equation for constant accel (bc using max accel for shortest stopping dist) added buffer dist to discourage speeds that are too fast
                    takes max and min between possible velocities robot can ever move and what the robot can acheive with current velocities and max accel
    prdeict_course(): finds path that is likely to be followed with set velocities over a time window given
    calc_scoring_vals(): finds scoring values for the path given distance to the obstacle (assumes circular obstacle and takes into account radius of obstacle and robot), difference in heading(within pi and 0), lastly distance to goal
    create_DWA_arcs(): normalizes scoring values and places clearance in a log so that it is given more importance the closer and closer it is to an obstacle, takes negative of the heading and distance scores bc the smaller the better for those, stores all possible arcs from ones provided and choses best scoring value, num_vel is hyperparam
        called before motion model to generate velocity
    
