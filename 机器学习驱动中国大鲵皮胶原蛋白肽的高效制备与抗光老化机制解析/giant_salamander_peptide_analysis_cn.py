import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

class GiantSalamanderPeptideAnalysis:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.ann_model = None
        self.rf_model = None
        self.ann_best_params = None
        self.rf_best_params = None
        
    def load_data(self, file_path):
        """加载数据"""
        self.data = pd.read_excel(file_path)
        print(f"数据加载完成，形状: {self.data.shape}")
        print("\n数据基本信息:")
        print(self.data.info())
        print("\n数据统计描述:")
        print(self.data.describe())
        
    def prepare_data(self, test_size=0.2, random_state=42):
        """准备训练和测试数据"""
        # 选择特征和目标变量
        feature_columns = ['水解时间_h', '酶剂量_U_per_g', '温度_°C', 'pH值', '固液比_w_v']
        self.X = self.data[feature_columns]
        self.y = self.data['弹性蛋白酶抑制率_EIR']
        
        # 分割训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # 标准化特征
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\n数据准备完成:")
        print(f"训练集: {self.X_train.shape}")
        print(f"测试集: {self.X_test.shape}")
        
    def train_ann_model(self):
        """训练人工神经网络模型"""
        print("\n开始训练人工神经网络(ANN)模型...")
        
        # 网格搜索优化参数
        param_grid = {
            'hidden_layer_sizes': [(9,), (12,), (9, 6), (12, 6)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'lbfgs'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [1000, 2000]
        }
        
        grid_search = GridSearchCV(
            MLPRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        self.ann_best_params = grid_search.best_params_
        self.ann_model = grid_search.best_estimator_
        
        print(f"\nANN模型最佳参数: {self.ann_best_params}")
        
    def train_rf_model(self):
        """训练随机森林模型"""
        print("\n开始训练随机森林(RF)模型...")
        
        # 网格搜索优化参数
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        self.rf_best_params = grid_search.best_params_
        self.rf_model = grid_search.best_estimator_
        
        print(f"\nRF模型最佳参数: {self.rf_best_params}")
        
    def evaluate_model(self, model, X_test, y_test, model_name):
        """评估模型性能"""
        y_pred = model.predict(X_test)
        
        metrics = {
            'R²': r2_score(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100
        }
        
        print(f"\n{model_name}模型性能评估:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
        return metrics, y_pred
    
    def feature_importance_analysis(self):
        """特征重要性分析"""
        if self.rf_model:
            feature_importance = self.rf_model.feature_importances_
            feature_names = self.X.columns
            
            importance_df = pd.DataFrame({
                '特征': feature_names,
                '重要性': feature_importance
            }).sort_values('重要性', ascending=False)
            
            print("\n特征重要性分析:")
            print(importance_df)
            
            return importance_df
        return None
    
    def create_comparison_plots(self, ann_metrics, rf_metrics, ann_pred, rf_pred):
        """创建模型比较图表"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('中国大鲵皮胶原蛋白肽酶解工艺优化模型比较', fontsize=16, fontweight='bold')
        
        # 1. 模型性能比较雷达图
        metrics = ['R²', 'RMSE', 'MAE', 'MAPE']
        ann_values = [ann_metrics['R²'], ann_metrics['RMSE'], ann_metrics['MAE'], ann_metrics['MAPE']]
        rf_values = [rf_metrics['R²'], rf_metrics['RMSE'], rf_metrics['MAE'], rf_metrics['MAPE']]
        
        # 标准化数据以便在雷达图上比较
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        all_values = np.array([ann_values, rf_values])
        scaled_values = scaler.fit_transform(all_values.T).T
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        scaled_values = np.concatenate((scaled_values, scaled_values[:, :1]), axis=1)
        angles += angles[:1]
        metrics += metrics[:1]
        
        ax1 = axes[0, 0]
        ax1.plot(angles, scaled_values[0], 'o-', linewidth=2, label='ANN模型', color='#2E86AB')
        ax1.fill(angles, scaled_values[0], alpha=0.25, color='#2E86AB')
        ax1.plot(angles, scaled_values[1], 'o-', linewidth=2, label='RF模型', color='#A23B72')
        ax1.fill(angles, scaled_values[1], alpha=0.25, color='#A23B72')
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(metrics[:-1])
        ax1.set_ylim(0, 1)
        ax1.set_title('模型性能雷达图', fontweight='bold')
        ax1.legend()
        ax1.grid(True)
        
        # 2. 预测值vs实际值散点图
        ax2 = axes[0, 1]
        ax2.scatter(self.y_test, ann_pred, alpha=0.6, label='ANN模型', color='#2E86AB', s=50)
        ax2.scatter(self.y_test, rf_pred, alpha=0.6, label='RF模型', color='#A23B72', s=50)
        ax2.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 
                'k--', lw=2, label='理想预测')
        ax2.set_xlabel('实际EIR值 (%)')
        ax2.set_ylabel('预测EIR值 (%)')
        ax2.set_title('预测值 vs 实际值', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 残差分析
        ax3 = axes[1, 0]
        ann_residuals = self.y_test - ann_pred
        rf_residuals = self.y_test - rf_pred
        
        ax3.scatter(ann_pred, ann_residuals, alpha=0.6, label='ANN模型', color='#2E86AB', s=50)
        ax3.scatter(rf_pred, rf_residuals, alpha=0.6, label='RF模型', color='#A23B72', s=50)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.8)
        ax3.set_xlabel('预测EIR值 (%)')
        ax3.set_ylabel('残差')
        ax3.set_title('残差分析', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 特征重要性
        ax4 = axes[1, 1]
        if self.rf_model:
            importance_df = self.feature_importance_analysis()
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            bars = ax4.barh(importance_df['特征'], importance_df['重要性'], color=colors)
            ax4.set_xlabel('重要性得分')
            ax4.set_title('特征重要性分析 (RF模型)', fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')
            
            # 在条形图上添加数值
            for bar in bars:
                width = bar.get_width()
                ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('model_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\n模型比较图表已保存为 model_comparison_analysis.png")
        
    def optimization_analysis(self):
        """工艺参数优化分析"""
        if self.ann_model:
            print("\n开始工艺参数优化分析...")
            
            # 创建参数网格用于优化分析
            param_ranges = {
                '水解时间_h': np.linspace(1, 8, 50),
                '酶剂量_U_per_g': np.linspace(2000, 10000, 50),
                '温度_°C': np.linspace(40, 80, 50),
                'pH值': np.linspace(6, 10, 50),
                '固液比_w_v': np.linspace(0.5, 3, 50)
            }
            
            # 单因素优化分析
            fig, axes = plt.subplots(3, 2, figsize=(15, 18))
            fig.suptitle('酶解工艺参数单因素优化分析', fontsize=16, fontweight='bold')
            
            for i, (param_name, param_values) in enumerate(param_ranges.items()):
                if i < 5:  # 只显示前5个参数
                    row, col = divmod(i, 2)
                    ax = axes[row, col]
                    
                    # 固定其他参数为平均值
                    base_params = self.X.mean().values
                    eir_predictions = []
                    
                    for value in param_values:
                        test_params = base_params.copy()
                        test_params[i] = value
                        test_params_scaled = self.scaler.transform([test_params])
                        prediction = self.ann_model.predict(test_params_scaled)[0]
                        eir_predictions.append(prediction)
                    
                    ax.plot(param_values, eir_predictions, 'b-', linewidth=2, color='#2E86AB')
                    ax.set_xlabel(param_name)
                    ax.set_ylabel('预测EIR值 (%)')
                    ax.set_title(f'{param_name}对EIR的影响', fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    # 标记最佳点
                    max_idx = np.argmax(eir_predictions)
                    ax.scatter(param_values[max_idx], eir_predictions[max_idx], 
                             color='red', s=100, zorder=5)
                    ax.annotate(f'最佳: {param_values[max_idx]:.2f}',
                               xy=(param_values[max_idx], eir_predictions[max_idx]),
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
            
            # 删除多余的子图
            axes[2, 1].remove()
            
            plt.tight_layout()
            plt.savefig('parameter_optimization_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("参数优化分析图表已保存为 parameter_optimization_analysis.png")
            
            # 多参数优化 - 寻找最佳组合
            from scipy.optimize import minimize
            
            def objective(params):
                # 负的EIR值作为目标函数（因为我们要最大化EIR）
                params_scaled = self.scaler.transform([params])
                return -self.ann_model.predict(params_scaled)[0]
            
            # 初始猜测值（使用平均值）
            initial_guess = self.X.mean().values
            
            # 参数边界
            bounds = [
                (1, 8),      # 水解时间
                (2000, 10000), # 酶剂量
                (40, 80),    # 温度
                (6, 10),     # pH值
                (0.5, 3)     # 固液比
            ]
            
            result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            
            optimal_params = result.x
            optimal_eir = -result.fun
            
            print(f"\n多参数优化结果:")
            print(f"最佳工艺参数组合:")
            for i, param_name in enumerate(self.X.columns):
                print(f"  {param_name}: {optimal_params[i]:.2f}")
            print(f"预测最佳EIR值: {optimal_eir:.2f}%")
            
            return optimal_params, optimal_eir
        return None, None
    
    def generate_report(self, ann_metrics, rf_metrics, optimal_params=None, optimal_eir=None):
        """生成分析报告"""
        report = f"""
# 中国大鲵皮胶原蛋白肽酶解工艺机器学习优化分析报告

## 1. 研究概述
本研究基于机器学习方法，优化中国大鲵皮胶原蛋白肽的酶解工艺参数，以提高其弹性蛋白酶抑制率（EIR），从而增强其抗光老化活性。

## 2. 数据概况
- 数据集大小: {self.data.shape[0]} 个样本，{self.data.shape[1]} 个变量
- 输入特征: 水解时间、酶剂量、温度、pH值、固液比
- 目标变量: 弹性蛋白酶抑制率（EIR）

## 3. 模型性能比较

### 3.1 人工神经网络（ANN）模型
- 最佳参数: {self.ann_best_params}
- R²: {ann_metrics['R²']:.4f}
- RMSE: {ann_metrics['RMSE']:.4f}
- MAE: {ann_metrics['MAE']:.4f}
- MAPE: {ann_metrics['MAPE']:.2f}%

### 3.2 随机森林（RF）模型
- 最佳参数: {self.rf_best_params}
- R²: {rf_metrics['R²']:.4f}
- RMSE: {rf_metrics['RMSE']:.4f}
- MAE: {rf_metrics['MAE']:.4f}
- MAPE: {rf_metrics['MAPE']:.2f}%

### 3.3 模型选择建议
基于R²值比较，{'ANN模型' if ann_metrics['R²'] > rf_metrics['R²'] else 'RF模型'}表现更优，
建议用于实际工艺参数优化。

## 4. 特征重要性分析
"""

        # 添加特征重要性分析
        importance_df = self.feature_importance_analysis()
        if importance_df is not None:
            report += "\n| 排名 | 特征 | 重要性得分 |\n"
            report += "|------|------|------------|\n"
            for i, row in importance_df.iterrows():
                report += f"| {len(importance_df)-i} | {row['特征']} | {row['重要性']:.4f} |\n"
        
        # 添加优化结果
        if optimal_params is not None:
            report += f"""
## 5. 工艺参数优化结果

### 5.1 最佳工艺参数组合
"""
            for i, param_name in enumerate(self.X.columns):
                report += f"- **{param_name}**: {optimal_params[i]:.2f}\n"
            
            report += f"""
### 5.2 预期效果
- 预测最佳EIR值: {optimal_eir:.2f}%
- 相比原始数据平均值提升: {((optimal_eir - self.y.mean()) / self.y.mean() * 100):.1f}%

## 6. 结论与建议

### 6.1 主要发现
1. 机器学习模型能够有效预测酶解工艺参数与EIR之间的关系
2. {'ANN' if ann_metrics['R²'] > rf_metrics['R²'] else 'RF'}模型在预测精度方面表现更优
3. 通过参数优化可以显著提高胶原蛋白肽的抗光老化活性

### 6.2 工艺优化建议
1. 按照优化后的参数进行酶解反应
2. 重点关注重要性较高的工艺参数
3. 建议进行实验验证以确认预测结果

### 6.3 后续研究方向
1. 扩大样本量以提高模型泛化能力
2. 考虑更多影响因素，如酶的种类组合、预处理方法等
3. 进行体外和体内实验验证抗光老化效果
"""
        
        # 保存报告
        with open('analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\n分析报告已生成并保存为 analysis_report.md")
        return report

# 主执行函数
def main():
    # 创建分析对象
    analyzer = GiantSalamanderPeptideAnalysis()
    
    # 1. 加载数据
    analyzer.load_data('simulated_giant_salamander_data.xlsx')
    
    # 2. 准备数据
    analyzer.prepare_data()
    
    # 3. 训练模型
    analyzer.train_ann_model()
    analyzer.train_rf_model()
    
    # 4. 评估模型
    ann_metrics, ann_pred = analyzer.evaluate_model(
        analyzer.ann_model, analyzer.X_test_scaled, analyzer.y_test, 'ANN'
    )
    
    rf_metrics, rf_pred = analyzer.evaluate_model(
        analyzer.rf_model, analyzer.X_test, analyzer.y_test, 'RF'
    )
    
    # 5. 创建可视化图表
    analyzer.create_comparison_plots(ann_metrics, rf_metrics, ann_pred, rf_pred)
    
    # 6. 参数优化分析
    optimal_params, optimal_eir = analyzer.optimization_analysis()
    
    # 7. 生成分析报告
    report = analyzer.generate_report(ann_metrics, rf_metrics, optimal_params, optimal_eir)
    
    print("\n" + "="*60)
    print("中国大鲵皮胶原蛋白肽酶解工艺优化分析完成！")
    print("="*60)
    print("生成的文件:")
    print("1. simulated_giant_salamander_data.xlsx - 模拟数据集")
    print("2. model_comparison_analysis.png - 模型比较图表")
    print("3. parameter_optimization_analysis.png - 参数优化图表")
    print("4. analysis_report.md - 详细分析报告")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()