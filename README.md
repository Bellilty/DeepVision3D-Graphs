
**GeoDeep3D-Graphs** is an advanced project demonstrating the application of **Geometric Deep Learning** to two critical areas:
1. **Point Cloud Classification**: Using a modified PointNet architecture for 3D object recognition.
2. **Graph Classification**: Employing a custom Graph Convolutional Network (GCN) for analyzing and classifying graph structures.

This project highlights state-of-the-art methods to solve real-world challenges in 3D data processing and graph analytics, showcasing innovative preprocessing, model design, and training strategies.

---

## Key Features

### 1. Point Cloud Classification
- **Dataset**: [ModelNet10](https://modelnet.cs.princeton.edu/) — a benchmark dataset of 3D CAD models spanning 10 categories.
- **Model**: A modified **PointNet** architecture, enhanced with T-Net for spatial alignment, batch normalization for stability, and dropout for regularization.
- **Results**: Achieved **90.09% test accuracy** with robust generalization to unseen data.

### 2. Graph Classification
- **Dataset**: Custom graph dataset containing node and edge attributes, provided as PyTorch Geometric (PyG) data objects.
- **Model**: A **Graph Convolutional Network (GCN)** enhanced with centrality-based features to better capture graph structure.
- **Results**: Achieved **83-86% validation accuracy**, leveraging innovative prediction refinement techniques.

---

## Real-World Applications
- **3D Object Classification**: Used in robotics, autonomous navigation, AR/VR, and CAD model recognition.
- **Graph Analytics**: Applications in network analysis, social media graphs, biological data, and more.

---

## Repository Structure

```
GeoDeep3D-GraphModels/
├── point-cloud-classification/
│   ├── src/                        # Codebase for PointNet model
│   │   ├── main.py                 # Training and evaluation script
│   │   ├── model.py                # PointNet architecture
│   │   ├── requirements1.txt        # Dependencies
│   │   ├── report.doc              # Task-specific report
├── graph-classification/
│   ├── src/                        # Codebase for GCN model
│   │   ├── main.py                 # Training and evaluation script
│   │   ├── model.py                # GCN architecture
│   │   ├── requirements2.txt        # Dependencies
│   │   ├── report.doc              # Task-specific report
│   │   ├── predictions.csv         # Test predictions
│   ├── results/                    # Reports and predictions
│   ├── README.md                   # Task-specific README
├── README.md                       # Main project README

```

---

## Getting Started

### Prerequisites
- Python 3.9+
- PyTorch and PyTorch Geometric

### Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/GeoDeep3D-GraphModels.git
cd GeoDeep3D-GraphModels
```

Install dependencies for each task:
```bash
pip install -r point-cloud-classification/src/requirements.txt
pip install -r graph-classification/src/requirements.txt
```

### Run the Code
**Point Cloud Classification**:
```bash
cd point-cloud-classification/src
python main.py
```

**Graph Classification**:
```bash
cd graph-classification/src
python main.py
```

---

## Results and Visualizations

### Point Cloud Classification
- **Examples of Correct Classifications**:
  Clear object features, such as chair legs or table tops, enabled accurate recognition.

- **Examples of Misclassifications**:
  Ambiguous or overlapping features caused errors, particularly between objects with similar structures.

![Correct Classification Example](results/success-examples/chair.png)

### Graph Classification
- **Validation Accuracy**: 83-86%
- **Prediction Refinement**: Used bootstrap methods and confidence scores to improve predictions for ambiguous graphs.

---

## Authors
- Simon Bellilty (345233563)
- Roni Fridman (205517097)



