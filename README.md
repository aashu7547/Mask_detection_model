# 😷 Face Mask Detection using CNN (TensorFlow & Keras)

This project implements a **Convolutional Neural Network (CNN)** to classify whether a person is **wearing a mask** or **not wearing a mask** from face images.  
The dataset used is the [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset).

---

## 📂 Dataset
The dataset contains two folders:
- `with_mask` → 690 images  
- `without_mask` → 686 images  

Total: **1376 images**

---

## ⚙️ Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/face-mask-detection.git
cd face-mask-detection

# Install dependencies
pip install tensorflow keras matplotlib pillow numpy scikit-learn kagglehub
🏗️ Model Architecture
Conv2D (32 filters, 3×3) + MaxPooling (2×2)

Conv2D (64 filters, 3×3) + MaxPooling (2×2)

Flatten

Dense (128, relu) + Dropout (0.5)

Dense (64, relu) + Dropout (0.5)

Dense (2, softmax) → Output: With Mask / Without Mask

Loss Function: sparse_categorical_crossentropy
Optimizer: Adam
Metrics: accuracy

🚀 Training
python
Copy
Edit
history = model.fit(
    X_train_scaled, 
    Y_train, 
    validation_split=0.1, 
    epochs=5
)
🔮 Prediction Example
python
Copy
Edit
import matplotlib.pyplot as plt

plt.imshow(X_test_scaled[10])
plt.axis("off")
plt.show()

prediction = model.predict(X_test_scaled[10].reshape(1,128,128,3)).argmax(axis=1)

if prediction[0] == 1:
    print("Prediction: WITH MASK")
else:
    print("Prediction: WITHOUT MASK")

print("Actual:", "WITH MASK" if Y_test[10] == 1 else "WITHOUT MASK")
📊 Results
Achieved high accuracy in distinguishing between masked and unmasked faces.

Can be extended for real-time mask detection using webcam feed (OpenCV).

📌 Future Improvements
Use Data Augmentation (rotation, flip, zoom) to improve generalization.

Train for more epochs and tune hyperparameters.

Deploy model in a web or mobile application.

👨‍💻 Author
Aashutosh Kumar

Dataset: Face Mask Dataset
