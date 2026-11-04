# CIFAR-10 Image Classification Solution

This project implements a complete, production-ready deep learning pipeline using **PyTorch** to classify images from the CIFAR-10 dataset. It also integrates **Residual Block** to improve the baseline model's performance.

---

## 📂 Project Structure

```text
cifar10_classifier/
├── data/                    # Automatically created; holds downloaded CIFAR-10 dataset
├── best_model.pth           # Saved weights of the top-performing model checkpoint
│── evaluation_curves.png    # Exported Training vs Validation loss & accuracy plots
├── cifar10_classifier.py    # Main Python script (Data pipeline, model architecture, training, evaluation)
├── requirements.txt         # Package dependencies list
└── README.md                # Project documentation and run guide
```

---

## 🏗️ Model Design Decisions

The network architecture transitions from a standard sequential Convolutional Neural Network (CNN) into a residual-driven topology to maximize representation accuracy while keeping parameter counts lightweight.

1. **Feature Extraction Blocks**: Consists of repeated convolutional layers utilizing $3 \times 3$ kernel dimensions to preserve spatial dimensions efficiently while mapping channels upwards from 3 (RGB) to 128 features.
2. **Residual Connections**: Incorporates a simplified Residual Block featuring identity skip-connections. If spatial or channel dimensions change, a $1 \times 1$ convolution adjusts them dynamically. This design mitigates the vanishing gradient problem, enabling faster convergence and deeper representation capacity.
3. **Regularization Framework**: 
   * **Batch Normalization (`BatchNorm2d`)**: Placed immediately after convolutional layers to stabilize internal covariate shift and accelerate training.
   * **Dropout**: Integrated right before the classification head ($p=0.3$) to reduce co-adaptation of features and combat overfitting.

---

## ⚙️ Training Setup

The training strategy leverages robust optimization configurations to ensure high generalization performance:

* **Optimizer**: `Adam` optimizer with a baseline learning rate of `1e-3` for fast, adaptive weight updates.
* **Loss Function**: `CrossEntropyLoss` since this is a multi-class categorization task.
* **Learning Rate Schedule**: `CosineAnnealingLR` scheduler that smoothly decays the learning rate to a target minimum, allowing the model to settle gracefully into sharp local minima near the end of training.
* **Checkpoint Optimization**: The pipeline continuously validates performance after every epoch, auto-exporting only the model state that achieves the lowest validation loss metrics to `best_model.pth`.

---

## 🚀 How to Run the Code

Follow these commands in your terminal or command prompt to execute the solution.

### 1. Setup Environment
Clone or navigate into your workspace folder and initialize an isolated virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Packages
Upgrade package management tools and batch install project library dependencies:
```bash
pip install --upgrade pip
pip install torch torchvision matplotlib
```

### 3. Trigger Training & Evaluation
Run the primary execution script. The code will handle dataset setup, run tracking loops, and output visual diagnostic curves automatically:
```
python cifar10_classifier.py
```
