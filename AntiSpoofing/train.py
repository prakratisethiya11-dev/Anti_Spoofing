from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

def main() :
    model.train(data='Dataset/SplitData/dataOffline.yaml', epochs=3)
   
if __name__ == "__main__":
    main()

