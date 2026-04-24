from controller import Supervisor   #type: ignore
import math

# Use Supervisor instead of Robot
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Get the ball node using DEF name
ball_node = robot.getFromDef("TENNIS_BALL")

if ball_node is None:
    print("Error: Could not find DEF node 'TENNIS_BALL'")
    exit(1)

# Movement Parameters
forward_speed = 0.2  # Constant speed on X axis
amplitude = 0.2      # How far it moves side-to-side (Z axis)
frequency = 0.2      # Oscillations per second

while robot.step(timestep) != -1:
    # Get the current simulation time
    t = robot.getTime()
    
    # Calculate lateral velocity (derivative of position)
    # To stay strictly on a sine path, we set the Z velocity:
    # vz = A * 2 * pi * f * cos(2 * pi * f * t)
    lat_velocity = amplitude * (2 * math.pi * frequency) * math.cos(2 * math.pi * frequency * t)
    
    # Velocity vector: [vx, vy, vz, wx, wy, wz]
    # We move forward on X and oscillate on Z
    velocity = [forward_speed, lat_velocity, 0, 0, 0, 0]
    
    ball_node.setVelocity(velocity)