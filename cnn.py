# %% Veri setini içeriye aktar ve preprocessing: normalisazyon, one-hot encoding

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10  # veri seti
from tensorflow.keras.utils import to_categorical  # encoding
from tensorflow.keras.models import Sequential  # Sirali model
from tensorflow.keras.layers import Conv2D, MaxPooling2D # feature extraction
from tensorflow.keras.layers import Flatten, Dense, Dropout # classification
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator # data agumentation

from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")

# load cifar10
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

# gorsellestirme
class_labels = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# bazi goruntuleri ve etiketlerini gorsellestir
fig, axes = plt.subplots(1, 5, figsize=(15,10))

for i in range(5):
    axes[i].imshow(x_train[i])
    label = class_labels[int(y_train[i])]
    axes[i].set_title(label)
    axes[i].axis("off")
    
plt.show()

# veri seti normalizasyonu 
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

# one-hot encoding 
y_train = to_categorical(y_train,10) # 10 class var o yüzden 10 yaziyoruz
y_test = to_categorical(y_test,10)  


# %% Veri aktarimi (Data Augmentation)

datagen= ImageDataGenerator(
    rotation_range = 20, # 20 dereceye kadar dondurme saglar 
    width_shift_range = 0.2, # goruntuyu yatayda %20 kaydırma
    height_shift_range = 0.2, # goruntuyu dikeyde %20 kaydirma
    shear_range = 0.2, # goruntu uzerinde kaydirma
    zoom_range = 0.2, # goruntuye zoom uygulama
    horizontal_flip = True, # goruntuyu yatayda ters cevirme
    fill_mode = "nearest" # bos alanlari doldurmak icin en yakin pixel degerlerini kullan
    )

datagen.fit(x_train) # data augmentation'u eğitim verileri üzerinde uygula


# %% Create, Compile and Train Model

#cnn modeli olustur (base model)
model=Sequential()

#Feature Extraction: CONV => RELU => CONV => RELU => POOL => DROPOUT
model.add(Conv2D(32, (3,3), padding = "same", activation = "relu", input_shape = x_train.shape[1:]))
model.add(Conv2D(32, (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25)) #baglantilarin %25'ini rastgele olarak kapat

#Feature Extraction: CONV => RELU => CONV => RELU => POOL => DROPOUT
model.add(Conv2D(64, (3,3), padding = "same", activation = "relu"))
model.add(Conv2D(62, (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

# Classification: FLATTEN, DENSE, RELU, DROPOUT, DENSE, (OUTPUT LAYER)
model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

model.summary()

# model derleme
model.compile(optimizer = RMSprop(learning_rate = 0.0001, decay = 1e-6),
              loss = "categorical_crossentropy",
              metrics = ["accuracy"])

# model training
history = model.fit(datagen.flow(x_train, y_train, batch_size = 512),
          epochs = 10,
          validation_data = (x_test, y_test)
          )


# %% Test model and evaluate performance

# modelin test seti uzerinden tahminini yap
y_pred = model.predict(x_test) # y_pred = [0, 1] mesela 0.8 çıkarsa %80 olma olasılıgı olarak tariflenebilir
y_pred_class = np.argmax(y_pred, axis = 1) # tamin edlen siniflari al
y_true=np.argmax(y_test, axis = 1)

# classification report hesapla
report = classification_report(y_true, y_pred_class, target_names = class_labels)
print(report)

plt.figure()

# kayip grafikleri
plt.subplot(1, 2, 1) # 1 satir, 2 sutun ve 1.subplot
plt.plot(history.history["loss"], label = "Train Loss")
plt.plot(history.history["val_loss"], label = "Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label = "Train Accuracy")
plt.plot(history.history["val_accuracy"], label = "Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()



























































