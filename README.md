Official implementation of "Correlation Guided Multi-Teacher Distillation for Lightweight Image Retrieval". 

# CGD: Correlation Guided Multi-Teacher Distillation for Lightweight Image Retrieval


Official PyTorch implementation of the paper **"Correlation Guided Multi-Teacher Distillation for Lightweight Image Retrieval"**.[Code coming soon]

## 🚀 Introduction
[Insert a brief introduction or abstract here...]

## 🛠️ Method
The overall framework of our proposed CGD method is illustrated below:

![Framework](assets/framework.png)
<div align="center">
  Figure 1: Overview of the Correlation Guided Multi-Teacher Distillation framework.
</div>

## 📊 Main Results
Our method achieves state-of-the-art performance on standard benchmarks while maintaining low computational cost.

### 📊 Main Results

Performance comparison on standard benchmarks (ROxford and RParis).


| Method | Backbone | Params (M) | ROxf (M) | ROxf (H) | RPar (M) | RPar (H) | ROxf+1M (M) | ROxf+1M (H) | RPar+1M (M) | RPar+1M (H) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ours†** | MobileNetV2 | 4.8 | 84.74 | 67.82 | 91.80 | 83.62 | 79.91 | 58.85 | 82.21 | 67.78 |
| **Ours†** | EfficientNetB3| 11.5 | 85.48 | 68.71 | 91.98 | 83.46 | 80.70 | 59.74 | 82.54 | 68.48 |
| | | | | | | | | | | |
| **Ours** | MobileNetV2 | 4.8 | 80.99 | 61.91 | 89.78 | 79.02 | 75.16 | 50.77 | 78.16 | 59.44 |
| **Ours** | EfficientNetB3| 11.5 | 82.38 | 64.37 | 90.44 | 80.19 | 77.83 | 53.63 | 81.31 | 63.59 | **[https://drive.google.com/drive/folders/1mAZppSYGbGe7OL-EnK0_uG13j2RYD1ls?usp=sharing]** |

> **Note:** The pre-trained weights can be downloaded from the links in the table above.

## 🔨 Installation
```bash
git clone [https://github.com/ban0329/CGD.git](https://github.com/ban0329/CGD.git)
cd CGD
pip install -r requirements.txt
