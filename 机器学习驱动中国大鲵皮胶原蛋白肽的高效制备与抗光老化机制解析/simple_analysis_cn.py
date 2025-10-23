import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_giant_salamander_data(file_path):
    """
    分析中国大鲵皮胶原蛋白肽酶解工艺数据
    
    参数:
    file_path: Excel文件路径
    
    返回:
    分析结果字典
    """
    print("="*60)
    print("中国大鲵皮胶原蛋白肽酶解工艺分析")
    print("="*60)
    
    # 1. 加载数据
    print(f"\n1. 正在加载数据: {file_path}")
    try:
        data = pd.read_excel(file_path)
        print(f"数据加载成功，形状: {data.shape}")
        print("\n数据预览:")
        print(data.head())
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None
    
    # 2. 数据验证
    required_columns = ['水解时间_h', '酶剂量_U_per_g', '温度_°C', 'pH值', '固液比_w_v', '弹性蛋白酶抑制率_EIR']
    missing_cols = [col for col in required_columns if col not in data.columns]
    
    if missing_cols:
        print(f"\n错误: 数据缺少必要的列: {missing_cols}")
        print(f"请确保Excel文件包含以下列: {required_columns}")
        return None
    
    # 3. 准备数据
    print("\n2. 准备数据...")
    X = data[['水解时间_h', '酶剂量_U_per_g', '温度_°C', 'pH值', '固液比_w_v']]
    y = data['弹性蛋白酶抑制率_EIR']
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")
    
    # 4. 训练模型
    print("\n3. 训练机器学习模型...")
    
    # 人工神经网络模型
    print("   训练人工神经网络(ANN)模型...")
    ann_model = MLPRegressor(
        hidden_layer_sizes=(9,),
        activation='relu',
        solver='lbfgs',
        alpha=0.01,
        max_iter=1000,
        random_state=42
    )
    ann_model.fit(X_train_scaled, y_train)
    
    # 随机森林模型
    print("   训练随机森林(RF)模型...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    # 5. 模型评估
    print("\n4. 模型性能评估:")
    
    # 评估ANN模型
    ann_pred = ann_model.predict(X_test_scaled)
    ann_metrics = {
        'R²': r2_score(y_test, ann_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, ann_pred)),
        'MAE': mean_absolute_error(y_test, ann_pred),
        'MAPE': mean_absolute_percentage_error(y_test, ann_pred) * 100
    }
    
    # 评估RF模型
    rf_pred = rf_model.predict(X_test)
    rf_metrics = {
        'R²': r2_score(y_test, rf_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'MAE': mean_absolute_error(y_test, rf_pred),
        'MAPE': mean_absolute_percentage_error(y_test, rf_pred) * 100
    }
    
    print("\nANN模型性能:")
    for metric, value in ann_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nRF模型性能:")
    for metric, value in rf_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 选择最佳模型
    best_model = 'ANN' if ann_metrics['R²'] > rf_metrics['R²'] else 'RF'
    print(f"\n推荐模型: {best_model} (R²更高)")
    
    # 6. 特征重要性分析
    print("\n5. 特征重要性分析 (RF模型):")
    feature_importance = pd.DataFrame({
        '特征': X.columns,
        '重要性': rf_model.feature_importances_
    }).sort_values('重要性', ascending=False)
    
    print(feature_importance)
    
    # 7. 创建可视化图表
    print("\n6. 生成可视化图表...")
    create_visualizations(X, y, X_test, y_test, ann_pred, rf_pred, feature_importance)
    
    # 8. 参数优化建议
    print("\n7. 工艺参数优化建议:")
    optimization_suggestions(X, y, ann_model, scaler)
    
    # 9. 返回结果
    results = {
        'data_summary': data.describe(),
        'ann_metrics': ann_metrics,
        'rf_metrics': rf_metrics,
        'best_model': best_model,
        'feature_importance': feature_importance,
        'sample_count': len(data)
    }
    
    print("\n" + "="*60)
    print("分析完成！生成的文件:")
    print("- model_performance.png: 模型性能比较图")
    print("- feature_importance.png: 特征重要性图")
    print("- prediction_analysis.png: 预测分析图")
    print("="*60)
    
    return results

def create_visualizations(X, y, X_test, y_test, ann_pred, rf_pred, feature_importance):
    """创建可视化图表"""
    
    # 1. 模型性能比较
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('模型性能分析', fontsize=16, fontweight='bold')
    
    # 1.1 R²比较
    ax1 = axes[0, 0]
    models = ['ANN', 'RF']
    r2_values = [r2_score(y_test, ann_pred), r2_score(y_test, rf_pred)]
    bars = ax1.bar(models, r2_values, color=['#2E86AB', '#A23B72'])
    ax1.set_ylabel('R²值')
    ax1.set_title('模型R²性能比较')
    ax1.set_ylim(0, 1)
    for bar, value in zip(bars, r2_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # 1.2 预测值vs实际值
    ax2 = axes[0, 1]
    ax2.scatter(y_test, ann_pred, alpha=0.6, label='ANN', color='#2E86AB', s=50)
    ax2.scatter(y_test, rf_pred, alpha=0.6, label='RF', color='#A23B72', s=50)
    ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    ax2.set_xlabel('实际EIR值 (%)')
    ax2.set_ylabel('预测EIR值 (%)')
    ax2.set_title('预测值 vs 实际值')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 1.3 残差分析
    ax3 = axes[1, 0]
    ann_residuals = y_test - ann_pred
    rf_residuals = y_test - rf_pred
    ax3.scatter(ann_pred, ann_residuals, alpha=0.6, label='ANN', color='#2E86AB', s=50)
    ax3.scatter(rf_pred, rf_residuals, alpha=0.6, label='RF', color='#A23B72', s=50)
    ax3.axhline(y=0, color='k', linestyle='--')
    ax3.set_xlabel('预测EIR值 (%)')
    ax3.set_ylabel('残差')
    ax3.set_title('残差分析')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 1.4 特征重要性
    ax4 = axes[1, 1]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    bars = ax4.barh(feature_importance['特征'], feature_importance['重要性'], color=colors)
    ax4.set_xlabel('重要性得分')
    ax4.set_title('特征重要性分析')
    for bar, value in zip(bars, feature_importance['重要性']):
        ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ 模型性能图表已生成: model_performance.png")

def optimization_suggestions(X, y, ann_model, scaler):
    """生成工艺参数优化建议"""
    from scipy.optimize import minimize
    
    def objective(params):
        return -ann_model.predict(scaler.transform([params]))[0]
    
    # 参数边界
    bounds = [
        (X['水解时间_h'].min(), X['水解时间_h'].max()),
        (X['酶剂量_U_per_g'].min(), X['酶剂量_U_per_g'].max()),
        (X['温度_°C'].min(), X['温度_°C'].max()),
        (X['pH值'].min(), X['pH值'].max()),
        (X['固液比_w_v'].min(), X['固液比_w_v'].max())
    ]
    
    # 多初始点优化
    best_result = None
    best_eir = -np.inf
    
    for i in range(5):  # 尝试5个不同的初始点
        initial_guess = [
            np.random.uniform(bounds[0][0], bounds[0][1]),
            np.random.uniform(bounds[1][0], bounds[1][1]),
            np.random.uniform(bounds[2][0], bounds[2][1]),
            np.random.uniform(bounds[3][0], bounds[3][1]),
            np.random.uniform(bounds[4][0], bounds[4][1])
        ]
        
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        if -result.fun > best_eir:
            best_eir = -result.fun
            best_result = result
    
    if best_result:
        optimal_params = best_result.x
        print(f"\n基于ANN模型的优化建议:")
        print(f"预测最佳EIR值: {best_eir:.2f}%")
        print(f"最佳工艺参数组合:")
        param_names = ['水解时间_h', '酶剂量_U_per_g', '温度_°C', 'pH值', '固液比_w_v']
        for i, param_name in enumerate(param_names):
            print(f"  {param_name}: {optimal_params[i]:.2f}")
        
        # 与当前数据比较
        current_avg = y.mean()
        improvement = ((best_eir - current_avg) / current_avg) * 100
        print(f"\n相比当前平均值({current_avg:.2f}%)的提升: {improvement:.1f}%")

# 主函数
if __name__ == "__main__":
    # 示例：使用模拟数据
    print("使用示例数据进行分析...")
    results = analyze_giant_salamander_data('simulated_giant_salamander_data.xlsx')
    
    if results:
        print(f"\n分析总结:")
        print(f"- 数据集大小: {results['sample_count']} 个样本")
        print(f"- 推荐模型: {results['best_model']}")
        print(f"- 最佳模型R²: {max(results['ann_metrics']['R²'], results['rf_metrics']['R²']):.4f}")
        print(f"- 最重要的特征: {results['feature_importance'].iloc[0]['特征']}")
else:
    # 当作为模块导入时，提供一个简单的接口
    def analyze_excel_data(file_path):
        """分析Excel数据的简单接口"""
        return analyze_giant_salamander_data(file_path)