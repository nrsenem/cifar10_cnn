# 🖼️ CIFAR-10 Image Classification with CNN

Bu proje, **CIFAR-10 veri seti** üzerinde bir **Convolutional Neural Network (CNN)** kullanarak görüntü sınıflandırma işlemi yapmaktadır.  
TensorFlow/Keras kütüphanesi kullanılarak **veri ön işleme, veri artırma (Data Augmentation), model oluşturma, eğitim ve değerlendirme** adımları gerçekleştirilmiştir.  

---

## 📂 Proje İçeriği
1. **Veri Seti ve Ön İşleme**
   - CIFAR-10 veri seti Keras üzerinden yüklenir.
   - Görseller [0,1] aralığına normalize edilir.
   - Etiketler one-hot encoding formatına dönüştürülür.
   - Örnek görseller matplotlib ile görselleştirilir.

2. **Data Augmentation**
   - Görsellerin çeşitliliğini artırmak için `ImageDataGenerator` ile:
     - Döndürme
     - Kaydırma
     - Zoom
     - Ayna çevirme (horizontal flip)  
     gibi işlemler uygulanır.

3. **CNN Modeli**
   - Katmanlar:
     - `Conv2D + ReLU` (özellik çıkarımı)
     - `MaxPooling2D` (boyut küçültme)
     - `Dropout` (overfitting’i engelleme)
     - `Flatten + Dense` (tam bağlı katmanlar)
     - `Softmax` (10 sınıf için çıktı)
   - Optimizer: **RMSprop**  
   - Loss: **categorical_crossentropy**

4. **Model Eğitimi**
   - `batch_size = 512`
   - `epochs = 10`
   - Eğitim sırasında train/validation kayıpları ve doğruluk oranları takip edilir.

5. **Değerlendirme**
   - Test verisi üzerinde tahminler alınır.
   - `classification_report` ile her sınıf için precision, recall, f1-score hesaplanır.
   - Eğitim sürecine ait loss ve accuracy grafiklendirilir.


### Gereksinimler
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

# 🖼️ CIFAR-10 Image Classification with CNN

This project implements an **image classification model** using a **Convolutional Neural Network (CNN)** on the **CIFAR-10 dataset**.  
The workflow includes **data preprocessing, data augmentation, model creation, training, and evaluation** with TensorFlow/Keras.  

---

## 📂 Project Content
1. **Dataset and Preprocessing**
   - Load CIFAR-10 dataset from Keras.
   - Normalize images to [0,1].
   - Convert labels to one-hot encoding.
   - Visualize sample images.

2. **Data Augmentation**
   - Enhance dataset variability with `ImageDataGenerator`:
     - Rotation
     - Shifting
     - Zoom
     - Horizontal flip  

3. **CNN Model**
   - Layers:
     - `Conv2D + ReLU` (feature extraction)
     - `MaxPooling2D` (downsampling)
     - `Dropout` (reduce overfitting)
     - `Flatten + Dense` (fully connected layers)
     - `Softmax` (output for 10 classes)
   - Optimizer: **RMSprop**  
   - Loss: **categorical_crossentropy**

4. **Model Training**
   - `batch_size = 512`
   - `epochs = 10`
   - Track training/validation accuracy and loss.

5. **Evaluation**
   - Predictions on the test set.
   - Generate `classification_report` with precision, recall, f1-score.
   - Plot accuracy and loss curves.

### Requirements
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn


📷 CIFAR-10 Classes (Sınıfları)
Airplane ✈️
Automobile 🚗
Bird 🐦
Cat 🐱
Deer 🦌
Dog 🐶
Frog 🐸
Horse 🐴
Ship 🚢
Truck 🚚

