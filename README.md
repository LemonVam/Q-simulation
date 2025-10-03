
# Quantum-enhanced Sentiment Analysis on IMDB

本项目结合了传统机器学习（Word2Vec + LSTM）与量子特征提取（Cirq），在IMDB电影评论数据集上进行情感分类研究。项目包含三大模块：

## 📂 文件结构

| 文件名 | 说明 |
|:------|:-----|
| `wwgpu_quantum_gridsearch.py` | 用于超参数（LSTM单元数、Dropout率）Grid Search优化 |
| `wwgpu_quantum_plot.py` | 可视化训练过程中 Loss 和 Accuracy 曲线 |
| `wwgpu_quantum_ablation.py` | 消融实验：比较 Word2Vec-only 和 Word2Vec+Quantum 特征模型性能 |

---

## 🛠️ 环境要求

- Python >= 3.8
- TensorFlow >= 2.10
- Cirq >= 1.0
- gensim
- matplotlib
- GPU可用 (建议)

安装依赖：
```bash
pip install tensorflow cirq gensim matplotlib
```

---

## 🚀 使用方法

### 1. 进行 Grid Search 超参数优化
```bash
python wwgpu_quantum_gridsearch.py
```
输出：最佳参数组合和对应测试集准确率。

### 2. 绘制 Loss/Accuracy 曲线
```bash
python wwgpu_quantum_plot.py
```
输出：保存 `LSTM_Word2Vec_training_curve.png`。

### 3. 进行消融实验 (Word2Vec vs Word2Vec+Quantum)
```bash
python wwgpu_quantum_ablation.py
```
输出：保存 `ablation_comparison.png`，对比两种特征方式在Validation上的表现。

---

## ⚠️ 注意事项
- 运行消融实验和Grid Search时，推荐使用GPU，否则训练速度可能较慢。
- 默认每个模型训练20个epoch，可以根据实际需要调整。
- Word2Vec是基于IMDB训练集动态训练的，并非预训练模型。
