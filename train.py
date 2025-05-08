# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from model.livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os


# xây dựng phân tích tham số đầu vào từ dòng lệnh
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", type=str, required=True, help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True, help="path to label encoder")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the initial learning rate, batch size, and number of epochs to train for
INIT_LR = 1e-4
BS = 8
EPOCHS = 50


# lấy danh sách các hình ảnh trong dicectory dataset, 
# sau đó khởi tạo danh sách dữ liệu (i.e., images) và label của class"
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []


# loop over all image paths
for imagePath in imagePaths:
	# trích xuất nhãn lớp từ dataset, tải hình ảnh và thay đổi 
    # kích thước nó thành 32x32 pixel, bỏ qua tỷ lệ frame
	label = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (32, 32))
	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)
# chuyển đổi dữ liệu thành mảng NumPy, sau đó tiền 
# xử lý bằng cách chuẩn hóa tất cả cường độ pixel về phạm vi [0, 1]
data = np.array(data, dtype="float") / 255.0

# mã hóa các nhãn (hiện tại là strings) thành số nguyên và sau đó mã hóa theo phương pháp one-hot 
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, 2)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# tăng cường dữ liệu cho training
# aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
#     width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
# 	horizontal_flip=True, fill_mode="nearest")

# --- Define Keras augmentation layers ---
data_augmentation_layers = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(32, 32, 3)),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(factor=0.055),  # rotation_range=20 degrees is approx 20/360 = 0.055 of 2*pi
    tf.keras.layers.RandomZoom(height_factor=0.15, width_factor=0.15), # zoom_range=0.15
    tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2), # width/height_shift_range
    # Note: shear_range is not directly available as a simple Keras layer
])

# --- Create tf.data.Dataset pipeline ---
def augment_data(image, label):
    image_expanded = tf.expand_dims(image, axis=0)
    augmented_image_batch = data_augmentation_layers(image_expanded, training=True)
    # Remove batch dimension before returning from map, so .batch(BS) works as expected
    augmented_image = tf.squeeze(augmented_image_batch, axis=0)
    return augmented_image, label

train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset = (
    train_dataset
    .shuffle(buffer_size=len(trainX))
    .map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BS)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

val_dataset = tf.data.Dataset.from_tensor_slices((testX, testY))
val_dataset = (
    val_dataset
    .batch(BS)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# initialize the optimizer and model
print("[INFO] compiling model...")
# opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
opt = Adam(learning_rate=INIT_LR) # Remove deprecated decay parameter
model = LivenessNet.build(width=32, height=32, depth=3, classes=len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network for {} epochs...".format(EPOCHS))
# H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS)
H = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

# save the network to disk
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])
# save the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()


# plot the training loss and accuracy
# Adjust the range for plotting if the number of epochs actually run differs
num_actual_epochs = len(H.history["loss"])

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, num_actual_epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, num_actual_epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, num_actual_epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, num_actual_epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])