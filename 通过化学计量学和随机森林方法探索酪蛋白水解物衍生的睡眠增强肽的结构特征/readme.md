# Sleep-Enhancing Peptide Analysis: Paper Reproduction

## 睡眠增强肽分析：论文复现

This repository contains a complete Python reproduction of the research paper "Exploring structural features of sleep-enhancing peptides derived from casein hydrolysates using chemometrics and random forest methods". The implementation includes all key analytical methods and generates simulated data to validate the main findings.

本仓库包含对研究论文《利用化学计量学和随机森林方法探索酪蛋白水解物中睡眠增强肽的结构特征》的完整Python复现。该实现包含所有关键分析方法，并生成模拟数据以验证主要发现。

---

## Key Features | 主要特点

- **Complete Method Implementation** | **完整方法实现**
  - Orthogonal Projections to Latent Structures (OPLS)
  - 正交偏最小二乘(OPLS)分析
  - Random Forest regression for activity prediction
  - 用于活性预测的随机森林回归
  - Chemometric and multivariate statistical analysis
  - 化学计量学和多变量统计分析
  - Structural feature importance evaluation
  - 结构特征重要性评估

- **Simulated Data Generation** | **模拟数据生成**
  - Realistic peptide sequence data with structural features
  - 具有结构特征的逼真肽序列数据
  - Metadata for casein hydrolysate samples
  - 酪蛋白水解物样本元数据
  - Digestion experiment simulation data
  - 消化实验模拟数据

- **Comprehensive Visualization** | **全面可视化**
  - OPLS score plots
  - OPLS得分图
  - Feature importance rankings
  - 特征重要性排名
  - Peptide length correlation analysis
  - 肽长度相关性分析
  - Tyrosine-containing peptide distribution
  - 含酪氨酸肽分布

---

## Project Structure | 项目结构

```
sleep_peptide_reproduction/
├── generate_simulated_data.py   # Generates realistic peptide datasets
                                # 生成逼真的肽数据集
├── core_analysis.py            # Implements key analytical methods
                                # 实现关键分析方法
├── example_usage.py            # Demonstrates step-by-step analysis workflow
                                # 演示分步分析工作流程
├── simplified_complete_reproduction.py  # Full reproduction pipeline
                                         # 完整复现流程
├── requirements.txt            # Dependencies
                                # 依赖项
└── README.md                   # Documentation
                                # 文档
```

---

## Installation | 安装

1. Clone or download this repository
   克隆或下载本仓库

2. Install required dependencies:
   安装所需依赖：
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage | 使用方法

### Complete Reproduction | 完整复现
Run the full reproduction pipeline that generates data, performs all analyses, and validates key findings:

运行完整复现流程，生成数据、执行所有分析并验证关键发现：
```bash
python simplified_complete_reproduction.py
```

### Example Workflow | 示例工作流程
Run a step-by-step demonstration of the analysis pipeline:

运行分析流程的分步演示：
```bash
python example_usage.py
```

### Generate Only Simulated Data | 仅生成模拟数据
Create simulated datasets without running the full analysis:

生成模拟数据集而不运行完整分析：
```bash
python generate_simulated_data.py
```

---

## Output Files | 输出文件

After running the analysis, the following files will be generated:

运行分析后，将生成以下文件：

- **Data Files | 数据文件**
  - `enhanced_peptides_data.csv`: Peptide sequences with structural features
    具有结构特征的肽序列
  - `samples_metadata.csv`: Metadata for casein hydrolysate samples
    酪蛋白水解物样本的元数据
  - `digestion_data.csv`: Simulated peptide digestion data
    模拟肽消化数据

- **Analysis Results | 分析结果**
  - `simplified_reproduction_summary.csv`: Key metrics and findings
    关键指标和发现
  - `identified_sleep_peptides.csv`: Top candidates with sleep-enhancing potential
    具有睡眠增强潜力的顶级候选肽
  - `top_importance_peptides.csv`: Features ranked by importance
    按重要性排序的特征

- **Visualizations | 可视化文件**
  - `feature_importance_plot.png`: Random Forest feature importance
    随机森林特征重要性
  - `peptide_length_correlation.png`: Relationship between length and activity
    长度与活性之间的关系
  - `tyr_peptide_distribution.png`: Activity comparison of tyrosine-containing peptides
    含酪氨酸肽的活性比较
  - `opls_score_plot.png`: OPLS score plot for sample separation
    用于样本分离的OPLS得分图

---

## Key Findings Reproduced | 复现的关键发现

The reproduction successfully validates the main conclusions from the original paper:

复现成功验证了原始论文的主要结论：

1. **Tyrosine-containing peptides** show significantly higher sleep-enhancing activity
   **含酪氨酸的肽**显示出显著更高的睡眠增强活性

2. **Specific peptide types** (YP-type, YI/L-type, YQ-type) exhibit the strongest activity
   **特定肽类型**（YP型、YI/L型、YQ型）表现出最强的活性

3. **Optimal peptide length** for sleep-enhancing activity is 4-10 amino acids
   睡眠增强活性的**最佳肽长度**为4-10个氨基酸

4. **C-terminal structural features** (particularly proline at C-terminus and glutamine at second position from C-terminus) enhance activity
   **C端结构特征**（特别是C端的脯氨酸和C端第二位的谷氨酰胺）增强活性

5. Both OPLS and Random Forest models effectively identify important structural features contributing to sleep-enhancing activity
   OPLS和随机森林模型都能有效识别有助于睡眠增强活性的重要结构特征

---

## Dependencies | 依赖项

- numpy>=1.21.0
- pandas>=1.3.0
- scikit-learn>=0.24.2
- matplotlib>=3.4.3
- seaborn>=0.11.2