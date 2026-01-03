"""
《模式识别与机器学习》课程大作业
学生考试通过预测系统
完整实现分类任务流程，包含：
1. 数据预处理
2. 特征工程
3. 模型训练（训练集80%）
4. 模型评估（测试集20%）
5. 结果分析
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    roc_auc_score,
    precision_recall_curve
)
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# 数据加载
def load_data():
    """生成符合实际场景的模拟数据集"""
    np.random.seed(42)
    n_samples = 1323
    
    # 基础特征
    data = {
        'age': np.random.randint(16, 55, size=n_samples),
        'prior_programming_experience': np.random.choice(
            ['None', 'Beginner', 'Intermediate', 'Advanced'], 
            size=n_samples,
            p=[0.2, 0.5, 0.2, 0.1]
        ),
        'weeks_in_course': np.random.randint(1, 16, size=n_samples),
        'hours_spent_learning_per_week': np.round(np.random.exponential(5, size=n_samples), 1),
        'practice_problems_solved': np.random.poisson(50, size=n_samples),
        'projects_completed': np.random.binomial(8, 0.3, size=n_samples),
        'passed_exam': np.random.binomial(1, 0.65, size=n_samples)
    }
    
    # 添加相关性特征
    df = pd.DataFrame(data)
    df['tutorial_videos_watched'] = df['hours_spent_learning_per_week'] * 3 + np.random.normal(0, 5, size=n_samples)
    df['debugging_sessions_per_week'] = df['projects_completed'] * 1.5 + np.random.poisson(2, size=n_samples)
    
    # 确保特征在合理范围内
    df['tutorial_videos_watched'] = df['tutorial_videos_watched'].clip(20, 60).astype(int)
    df['debugging_sessions_per_week'] = df['debugging_sessions_per_week'].clip(1, 15).astype(int)
    df['self_reported_confidence_python'] = (df['prior_programming_experience'].map(
        {'None':1, 'Beginner':3, 'Intermediate':7, 'Advanced':9}) + 
        np.random.randint(-2, 3, size=n_samples)).clip(1, 10)
    
    return df

# 数据预处理
def preprocess_data(df):
    """完整的数据清洗和特征工程流程"""
    df_clean = df.copy()
    
    # 1. 缺失值处理
    numeric_cols = ['age', 'weeks_in_course', 'hours_spent_learning_per_week',
                   'practice_problems_solved', 'projects_completed', 
                   'tutorial_videos_watched', 'debugging_sessions_per_week',
                   'self_reported_confidence_python']
    
    imputer = SimpleImputer(strategy='median')
    df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
    
    # 2. 分类变量编码
    le = LabelEncoder()
    df_clean['prior_programming_experience'] = le.fit_transform(
        df_clean['prior_programming_experience'])
    
    # 3. 特征工程
    df_clean['total_learning_hours'] = df_clean['weeks_in_course'] * df_clean['hours_spent_learning_per_week']
    df_clean['problem_solving_rate'] = df_clean['practice_problems_solved'] / df_clean['weeks_in_course']
    df_clean['project_intensity'] = df_clean['projects_completed'] / df_clean['weeks_in_course']
    
    # 4. 特征选择
    features = [
        'age', 'prior_programming_experience', 'weeks_in_course',
        'hours_spent_learning_per_week', 'practice_problems_solved',
        'projects_completed', 'tutorial_videos_watched', 
        'debugging_sessions_per_week', 'self_reported_confidence_python',
        'total_learning_hours', 'problem_solving_rate', 'project_intensity'
    ]
    
    X = df_clean[features]
    y = df_clean['passed_exam']
    
    return X, y, features

# 模型训练与评估
def train_model(X, y):
    """完整的模型训练和评估流程"""
    # 1. 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. 数据集划分 (训练集80%，测试集20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"\n数据集划分:")
    print(f"- 训练集样本数: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"- 测试集样本数: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # 3. 模型初始化
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    
    # 4. 模型训练
    model.fit(X_train, y_train)
    
    # 5. 模型评估
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n=== 模型评估结果 ===")
    print("准确率:", accuracy_score(y_test, y_pred))
    print("AUC得分:", roc_auc_score(y_test, y_prob))
    print("\n分类报告:\n", classification_report(y_test, y_pred))
    
    # 6. 交叉验证
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    print("\n交叉验证准确率:", cv_scores.mean())
    
    return model, X_test, y_test, y_prob

# 结果可视化
def visualize_results(model, features, X_test, y_test, y_prob):
    """完整的结果可视化"""
    plt.figure(figsize=(15, 6))
    
    # 1. 特征重要性
    plt.subplot(1, 2, 1)
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(data=importance.head(8), x='importance', y='feature')
    plt.title('Top 8 重要特征')
    
    # 2. PR曲线
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall 曲线')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 3. 混淆矩阵
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['未通过', '通过'],
                yticklabels=['未通过', '通过'])
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()

# 主函数
def main():
    """完整的工作流程"""
    print("=== 学生考试通过预测系统 ===")
    
    # 1. 数据加载
    print("\n[1/4] 数据加载...")
    df = load_data()
    print("数据集信息:")
    print("- 样本数:", len(df))
    print("- 初始特征数:", len(df.columns) - 1)
    print("- 通过率: {:.1f}%".format(df['passed_exam'].mean() * 100))
    
    # 2. 数据预处理
    print("\n[2/4] 数据预处理...")
    X, y, features = preprocess_data(df)
    print("- 处理后特征数:", len(features))
    
    # 3. 模型训练与评估
    print("\n[3/4] 模型训练与评估...")
    model, X_test, y_test, y_prob = train_model(X, y)
    
    # 4. 结果可视化
    print("\n[4/4] 结果可视化...")
    visualize_results(model, features, X_test, y_test, y_prob)

if __name__ == "__main__":
    main()
