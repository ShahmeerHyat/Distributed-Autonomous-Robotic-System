# controller.py
from controller import Robot
import zmq

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# ZeroMQ setup
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:5555")

# Example device
# sensor = robot.getDevice("distance sensor")
# sensor.enable(timestep)

while robot.step(timestep) != -1:
    
    # Example data (replace with real sensor reading)
    sensor_value = 42
    
    # Send data
    socket.send_json({"sensor": sensor_value})
    
    # Wait for reply
    reply = socket.recv_json()
    control = reply["control"]
    
    print("Control from server:", control)

    # Apply control to motors here