import numpy as np

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def heading_diff_calc(position, desired):
    vector = desired - position[:2]
    angle_needed = np.arctan2(vector[1], vector[0])
    return wrap_angle(angle_needed - position[2])

test_cases = [
    # Facing right, goal is up => +90°
    (np.array([0, 0, 0]), np.array([0, 1]), 90),

    # Facing right, goal is left => ±180°
    (np.array([0, 0, 0]), np.array([-1, 0]), 180),

    # Facing up, goal is right => -90°
    (np.array([0, 0, np.pi/2]), np.array([1, 0]), -90),

    # Facing up, goal is up-right => -45°
    (np.array([0, 0, np.pi/2]), np.array([1, 1]), -45),

    # Facing 45°, goal is 45° => 0°
    (np.array([0, 0, np.pi/4]), np.array([1, 1]), 0),

    # Facing 180°, goal is 0° => -180°
    (np.array([0, 0, np.pi]), np.array([1, 0]), -180),

    # Facing 0°, goal is southwest => -135°
    (np.array([0, 0, 0]), np.array([-1, -1]), -135),
]

for i, (position, desired, expected_deg) in enumerate(test_cases):
    diff_rad = heading_diff_calc(position, desired)
    diff_deg = np.rad2deg(diff_rad)
    print(f"Test {i+1}: Heading diff = {diff_deg:.1f}°, Expected ≈ {expected_deg}°")
