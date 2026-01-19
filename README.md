CIFAR-100 Image Classification

This project implements an image classification model trained on the CIFAR-100 dataset using deep learning techniques. The goal is to accurately classify images into one of 100 fine-grained object categories.

📌 About CIFAR-100

CIFAR-100 is a widely used computer vision benchmark dataset consisting of 60,000 color images (32×32) divided into 100 classes, with 600 images per class. It is commonly used to evaluate image classification and representation learning models.

Dataset Details

Total images: 60,000

Training images: 50,000

Test images: 10,000

Image size: 32 × 32 (RGB)

Classes: 100 fine labels (grouped into 20 coarse categories)

The dataset is automatically loaded using deep learning libraries.

Preprocessing

Pixel normalization to range [0, 1]

One-hot encoding of labels

Train–test split provided by the dataset

Optional data augmentation for better generalization

Model Architecture

Convolutional Neural Network (CNN)

Multiple Conv + ReLU layers

MaxPooling for downsampling

Fully connected Dense layers

Softmax activation for multi-class classification

The architecture is designed to balance performance and computational efficiency.

Results

Achieved strong classification accuracy on the test set

Stable training with minimal overfitting

Evaluation performed using accuracy metrics

Exact results may vary depending on training configuration and hardware.

How to Run

Open the notebook:

Final_Cifar10_85_percent_accuracy.ipynb


Run all cells sequentially

Training and evaluation will execute automatically

Compatible with Google Colab, Kaggle, and local Jupyter Notebook environments.

Requirements

Python 3.x

TensorFlow / Keras

NumPy

Matplotlib

Install dependencies using:

pip install tensorflow numpy matplotlib

🔮 Future Improvements

Experiment with ResNet or EfficientNet architectures

Apply advanced data augmentation techniques

Hyperparameter tuning

Transfer learning for improved accuracy

License

This project is intended for educational and research purposes.
