from ultralytics import YOLO

yolo_model = YOLO("yolov8m.pt")
yolo_model.export(format="openvino", imgsz=480)

yolo_model = YOLO("yolov8s.pt")
yolo_model.export(format="openvino", imgsz=480)

yolo_model = YOLO("yolov8n.pt")
yolo_model.export(format="openvino", imgsz=480)
