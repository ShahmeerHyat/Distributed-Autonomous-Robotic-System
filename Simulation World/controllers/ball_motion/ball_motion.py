from controller import Supervisor

# Use Supervisor instead of Robot
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Get the ball node using DEF name (must match DEF in .wbt file)
ball_node = robot.getFromDef("TENNIS_BALL")

if ball_node is None:
    print("Error: Could not find DEF node 'TENNIS_BALL'")
    exit(1)

# Constant forward velocity [vx, vy, vz, wx, wy, wz]
velocity = [0.2, 0, 0, 0, 0, 0]

while robot.step(timestep) != -1:
    ball_node.setVelocity(velocity)