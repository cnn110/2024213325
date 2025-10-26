import math
import random
def pearson_correlation(x, y):
  
    n = len(x)
    if len(y) != n:
        raise ValueError("x和y长度必须一致")

    
    mean_x = sum(x) / n
    mean_y = sum(y) / n

   
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator = math.sqrt(sum((xi - mean_x) ** 2 for xi in x)) * math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

    r = numerator / denominator if denominator != 0 else 0.0

    
    t_stat = r * math.sqrt((n - 2) / (1 - r ** 2)) if (1 - r ** 2) != 0 else 0.0
    
    p_value = 0.001 if abs(t_stat) > 3 else 0.05 if abs(t_stat) > 2 else 0.1
    return r, p_value



def t_test_ind(group1, group2):
   
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = sum(group1) / n1, sum(group2) / n2

    
    var1 = sum((x - mean1) ** 2 for x in group1) / (n1 - 1)
    var2 = sum((x - mean2) ** 2 for x in group2) / (n2 - 1)

  
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)

   
    t_stat = (mean1 - mean2) / math.sqrt(pooled_var * (1 / n1 + 1 / n2))

    
    df = n1 + n2 - 2  # 自由度
    p_value = 0.01 if abs(t_stat) > 2.5 else 0.05 if abs(t_stat) > 2 else 0.1
    return t_stat, p_value



def anova_f_value(*groups):
    """计算单因素ANOVA的F值和P值（简化版）"""
    k = len(groups)  # 组数
    n = sum(len(group) for group in groups)  # 总样本数

    # 总均值
    total_mean = sum(sum(group) for group in groups) / n

    # 组间平方和（SSB）和组内平方和（SSW）
    ssb = sum(len(g) * (sum(g) / len(g) - total_mean) ** 2 for g in groups)
    ssw = sum(sum((x - sum(g) / len(g)) ** 2 for x in g) for g in groups)

    # 均方
    msb = ssb / (k - 1)  # 组间均方
    msw = ssw / (n - k)  # 组内均方

    # F值
    f_stat = msb / msw if msw != 0 else 0.0

    # 简化P值计算（基于F分布模拟）
    p_value = 0.01 if f_stat > 5 else 0.05 if f_stat > 3 else 0.1
    return f_stat, p_value

def calculate_auc(y_true, y_pred_proba):
    """计算AUC值（基于排序法）"""
    # 合并真实标签和预测概率并按概率排序
    sorted_pairs = sorted(zip(y_pred_proba, y_true), key=lambda x: x[0])
    y_pred_sorted = [p for p, _ in sorted_pairs]
    y_true_sorted = [t for _, t in sorted_pairs]

    n_pos = sum(y_true)  # 正样本数
    n_neg = len(y_true) - n_pos  # 负样本数
    if n_pos == 0 or n_neg == 0:
        return 0.5  # 只有一类样本时AUC为0.5

    # 计算所有正样本在负样本之后的概率
    auc = 0.0
    neg_count = 0  # 记录当前负样本数量

    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 0:
            neg_count += 1
        else:
            auc += neg_count  # 正样本前的负样本数都算一次正确排序

    # 标准化到0-1范围
    auc /= (n_pos * n_neg)
    return auc


# ------------------------------
# 示例数据与计算
# ------------------------------
if __name__ == "__main__":
    # 相关分析数据
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # t检验数据
    group1 = [12, 13, 14, 15, 16]
    group2 = [18, 19, 20, 21, 22]

    # ANOVA数据
    group_a = [5, 6, 7, 8, 9]
    group_b = [7, 8, 9, 10, 11]
    group_c = [10, 11, 12, 13, 14]

    # AUC数据（真实标签和预测概率）
    y_true = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
    y_pred_proba = [0.2, 0.1, 0.1, 0.8, 0.8, 0.9, 0.6, 0.4, 0.7, 0.2]  # 手动设定的预测概率

    # 计算指标
    r, p_corr = pearson_correlation(x, y)
    t_stat, p_ttest = t_test_ind(group1, group2)
    f_stat, p_anova = anova_f_value(group_a, group_b, group_c)
    auc = calculate_auc(y_true, y_pred_proba)

    # 输出结果
    print("===== 手动计算统计指标结果 =====")
    print(f"1. 相关系数R: {r:.4f} (P值: {p_corr:.4f})")
    print(f"2. t检验P值: {p_ttest:.4f} (t统计量: {t_stat:.4f})")
    print(f"3. ANOVA F值: {f_stat:.4f} (P值: {p_anova:.4f})")
    print(f"4. AUC值: {auc:.4f}")












