from controller import Robot, Camera
import zmq
import msgpack
from PIL import Image
import io
import time

robot = Robot()
timestep = int(robot.getBasicTimeStep())

cam = robot.getDevice("camera") # Must match the name in Webots
cam.enable(timestep)

ctx = zmq.Context()
sock = ctx.socket(zmq.PUB)
sock.bind("tcp://*:5555")

print("initialzied")

while robot.step(timestep) != -1:
    # 1. Get raw BGRA bytes from Webots
    
    raw_image = cam.getImage()
    width = cam.getWidth()
    height = cam.getHeight()

    if raw_image:
        # 2. Convert BGRA to a Pillow Image
        # Webots gives BGRA; Pillow 'RGBA' expects RGBA. 
        # We can use Image.frombytes and then swap channels if needed
        img = Image.frombytes("RGBA", (width, height), raw_image, "raw", "BGRA")
        buf = io.BytesIO()
        # print("cURRENT TIME: ", timestep)
        img.convert("RGB").save(buf, format="JPEG")
        payload = {
            "time": robot.getTime(),
            "data": buf.getvalue()
        }
        sock.send(msgpack.packb(payload, use_bin_type=True))
        time.sleep(5)