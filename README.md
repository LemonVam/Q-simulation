# Quantum-Enhanced NLP via CPU-Based Bloch Simulation  
量子增强自然语言处理：基于 CPU 的 Bloch 球模拟  
CPUベースのブロッホ球シミュレーションによる量子強化型自然言語処理  

---

## 1. Overview / 概要 / 概要  

**English:**  
This project implements a hybrid *Quantum-Classical Sentiment Analysis* pipeline using the IMDB movie review dataset. It combines Word2Vec embeddings with simulated quantum feature extraction on the Bloch sphere (via Cirq) and evaluates classification performance using Logistic Regression and LDA models on CPU.  

**中文：**  
本项目实现了一个混合式「量子-经典」情感分析流程，使用 IMDB 影评数据集。代码通过 Word2Vec 生成文本向量，并在 Bloch 球上进行量子特征提取（基于 Cirq 模拟器），再利用逻辑回归与线性判别分析（LDA）在 CPU 上进行分类性能评估。  

**日本語：**  
本プロジェクトは、IMDB映画レビュー・データセットを用いたハイブリッド型の量子古典感情分析パイプラインです。Word2Vecで文を数値化し、Cirqによるブロッホ球上の量子特徴抽出を行い、ロジスティック回帰およびLDAモデルによりCPU上で分類性能を評価します。  

---

## 2. Environment Setup / 环境配置 / 環境構築  

**Requirements:**  
- Python ≥ 3.9  
- TensorFlow ≥ 2.13  
- gensim ≥ 4.3  
- cirq ≥ 1.2  
- scikit-learn ≥ 1.3  
- matplotlib ≥ 3.7  

**Install dependencies:**  
```bash
pip install tensorflow gensim cirq scikit-learn matplotlib
```

**中文说明：**  
本脚本仅依赖常规 CPU 环境，不需要量子硬件。安装上述依赖即可运行。  

**日本語：**  
量子ハードウェアは不要で、CPU環境で実行可能です。上記のライブラリをインストールしてください。  

---

## 3. How to Run / 运行方法 / 実行方法  

**English:**  
Run the script directly:
```bash
python cpuq.py
```
The program will:  
1. Load IMDB dataset  
2. Train a Word2Vec model  
3. Generate quantum states and Bloch vectors (X, Y, Z)  
4. Train classical classifiers (Logistic Regression, LDA)  
5. Output accuracy, loss, and visualizations  

**中文：**  
运行：
```bash
python cpuq.py
```
程序主要步骤：  
1. 加载 IMDB 数据集  
2. 使用 Word2Vec 训练词向量模型  
3. 生成量子态并提取 Bloch 球特征 (X, Y, Z)  
4. 使用逻辑回归与 LDA 进行分类  
5. 输出准确率、损失与可视化图像  

**日本語：**  
実行：
```bash
python cpuq.py
```
実行手順：  
1. IMDBデータセットを読み込み  
2. Word2Vecモデルを学習  
3. 量子状態を生成し、ブロッホ球の特徴（X, Y, Z）を抽出  
4. ロジスティック回帰とLDAで分類を実行  
5. 精度、損失、可視化結果を出力  

---

## 4. Output Files / 输出文件 / 出力ファイル  

After execution, several result files will be saved in the working directory:  

| File name | Description | 中文说明 | 日本語説明 |
|------------|--------------|----------|-------------|
| `comparison_accuracy.png` | Accuracy comparison (classical vs quantum) | 经典与量子模型的准确率对比图 | 古典モデルと量子モデルの精度比較 |
| `comparison_loss.png` | Loss comparison plot | 损失曲线比较图 | 損失曲線の比較 |
| `bloch_scatter_xyz_3d.png` | 3D Bloch sphere visualization | Bloch 球三维分布图 | 3次元ブロッホ球の可視化 |
| `bloch_violin_x.png` etc. | Feature distribution plots | Bloch 球 X/Y/Z 特征分布 | ブロッホ球のX/Y/Z分布 |
| `test_accuracy_bar.png` | Test accuracy bar chart | 测试集准确率柱状图 | テスト精度の棒グラフ |

---

## 5. Project Structure / 项目结构 / プロジェクト構成  

```
cpuq.py                     # Main script (quantum feature simulation & classification)
README.md                   # Documentation (this file)
└── output/ (optional)      # Stores generated figures and result data
```

**中文：**  
`cpuq.py` 是主脚本，负责从文本到量子特征再到分类的全过程。所有可视化图像会自动保存在当前目录。  

**日本語：**  
`cpuq.py` はメインスクリプトであり、テキスト処理から量子特徴抽出、分類までを実行します。結果図はカレントディレクトリに保存されます。  
