from click.core import batch

from ultralytics import YOLO

model = YOLO(r'D:\py_yuge\ultralytics-main\yolo11n.pt')
# Train the model
if __name__ == '__main__':

    model.train(
        data = r'D:\py_yuge\ultralytics-main\data.yaml',  # path to dataset YAML
        epochs = 250,  # number of training epochs
        batch = 4,
        imgsz = 640,  # training image size
        device = "cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        optimizer = 'SGD',
        project = 'runs/train',
        name = 'exp',
        single_cls = False,
        cache = False
    )