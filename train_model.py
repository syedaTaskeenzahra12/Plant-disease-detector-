"""
Train a simple CNN on the PlantVillage tomato subset.

Usage:
python train_model.py --epochs 15
"""
import os, argparse, tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--img_size", type=int, default=128)
args = parser.parse_args()

train_dir = "data/train"
if not os.path.isdir(train_dir):
    raise SystemExit("✖ Data not found. Run download_dataset.py first.")

datagen = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(args.img_size, args.img_size),
    batch_size=args.batch_size,
    subset="training"
)
val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(args.img_size, args.img_size),
    batch_size=args.batch_size,
    subset="validation"
)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(args.img_size, args.img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.3),
    Dense(256, activation="relu"),
    Dense(train_gen.num_classes, activation="softmax")
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

os.makedirs("model", exist_ok=True)
ckpt = ModelCheckpoint("model/plant_disease_model.h5",
                       save_best_only=True, monitor="val_accuracy", mode="max")
early = EarlyStopping(patience=3, restore_best_weights=True)

model.fit(train_gen,
          validation_data=val_gen,
          epochs=args.epochs,
          callbacks=[ckpt, early])
