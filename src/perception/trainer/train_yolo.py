from ultralytics import YOLO

# 训练前请准备 datasets/SAR/images/{train,val} 与 labels/{train,val}
# 并在 sar.yaml 中正确配置路径
def main():
    model = YOLO("yolov8n.pt")
    model.train(data="src/perception/trainer/sar.yaml", epochs=50, imgsz=640, batch=8)

if __name__ == "__main__":
    main()
