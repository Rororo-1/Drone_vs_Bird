import os
import glob
import shutil
import yaml
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from ultralytics import YOLO


drone_images = glob.glob("dataset/images_drones/*.png")
drone_labels = glob.glob("dataset/lables_drones/*.txt")

bird_images = glob.glob("dataset/images_birds/*.png")
bird_labels = glob.glob("dataset/lables_birds/*.txt")

drone_images.sort()
drone_labels.sort()

bird_images.sort()
bird_labels.sort()

d_train_im, d_test_im, d_train_lb, d_test_lb = train_test_split(drone_images, drone_labels, train_size=0.9, random_state=1)
d_train_im, d_val_im, d_train_lb, d_val_lb = train_test_split(d_train_im, d_train_lb, train_size=0.8, random_state=1)

b_train_im, b_test_im, b_train_lb, b_test_lb = train_test_split(bird_images, bird_labels, train_size=0.9, random_state=1)
b_train_im, b_val_im, b_train_lb, b_val_lb = train_test_split(b_train_im, b_train_lb, train_size=0.8, random_state=1)

train_im = d_train_im + b_train_im
train_lb = d_train_lb + b_train_lb

val_im = d_val_im + b_val_im
val_lb = d_val_lb + b_val_lb

test_im = d_test_im + b_test_im
test_lb = d_test_lb + b_test_lb

train_im, train_lb = shuffle(train_im, train_lb, random_state=1)
val_im, val_lb = shuffle(val_im, val_lb, random_state=1)
test_im, test_lb = shuffle(test_im, test_lb, random_state=1)

past = ['train/images', 'train/labels', 'val/images', 'val/labels', 'test/images', 'test/labels']

for p in past:
    os.makedirs(os.path.join('dataset/sort', p), exist_ok=True)

def coprit(dst_dir, files):
    for file_path in files:
        filename = os.path.basename(file_path)
        shutil.copy(file_path, os.path.join(dst_dir, filename))



coprit(os.path.join('dataset/sort/train/images'), train_im)
coprit(os.path.join('dataset/sort/train/labels'), train_lb)

coprit(os.path.join('dataset/sort/val/images'), val_im)
coprit(os.path.join('dataset/sort/val/labels'), val_lb)

coprit(os.path.join('dataset/sort/test/images'), test_im)
coprit(os.path.join('dataset/sort/test/labels'), test_lb)

data = {
    'path': os.path.abspath('dataset/sort'),
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images',
    'nc': 2,
    'names': ["Drone", "Bird"]
}
with open('dataset/data_yaml.yaml', 'w', encoding='utf-8') as file:
    yaml.dump(data, file, sort_keys=False)

model = YOLO("yolov8n.pt")

fit = model.train(data='dataset/data_yaml.yaml', epochs=30, imgsz=640, batch=16)
pred = model.predict(source='dataset/sort/test/images', save=True)

model = YOLO("runs/detect/train/weights/best.pt")
metrics = model.val(data='dataset/data_yaml.yaml', split='test')
print(metrics.box.map)      
print(metrics.box.map50)    
print(metrics.box.mp)      
print(metrics.box.mr)       