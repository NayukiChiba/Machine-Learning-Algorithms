"""
Scikit-learn 特征工程
对应文档: ../docs/03_feature_engineering.md

使用方式：
    from code.03_feature_engineering import *
    demo_text_vectorizer()
    demo_feature_selection()
"""

import numpy as np
from sklearn import datasets


def demo_count_vectorizer():
    """CountVectorizer 词频统计"""
    print("=" * 50)
    print("1. CountVectorizer 词频统计")
    print("=" * 50)
    
    from sklearn.feature_extraction.text import CountVectorizer
    
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
    ]
    
    cv = CountVectorizer()
    X = cv.fit_transform(corpus)
    
    print(f"词汇表: {cv.get_feature_names_out()}")
    print(f"词频矩阵:\n{X.toarray()}")
    print(f"稀疏矩阵类型: {type(X)}")


def demo_tfidf_vectorizer():
    """TfidfVectorizer 详解"""
    print("=" * 50)
    print("2. TfidfVectorizer")
    print("=" * 50)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
    ]
    
    # 基础用法
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(corpus)
    print(f"基础用法 - 特征数: {X.shape[1]}")
    
    # 常用参数
    tfidf_adv = TfidfVectorizer(
        max_features=100,      # 最多保留N个词
        min_df=1,              # 最少出现在N个文档
        max_df=0.9,            # 最多出现在90%文档
        ngram_range=(1, 2),    # 使用1-gram和2-gram
        stop_words='english'   # 去停用词
    )
    X_adv = tfidf_adv.fit_transform(corpus)
    print(f"高级用法 - 特征数: {X_adv.shape[1]}")
    print(f"词汇表: {tfidf_adv.get_feature_names_out()}")


def demo_dict_vectorizer():
    """DictVectorizer 字典特征提取"""
    print("=" * 50)
    print("3. DictVectorizer")
    print("=" * 50)
    
    from sklearn.feature_extraction import DictVectorizer
    
    data = [
        {'city': '北京', 'temperature': 20},
        {'city': '上海', 'temperature': 25},
        {'city': '北京', 'temperature': 18},
    ]
    
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(data)
    
    print(f"特征名: {dv.get_feature_names_out()}")
    print(f"特征矩阵:\n{X}")
    
    # 逆变换
    data_back = dv.inverse_transform(X)
    print(f"\n逆变换: {data_back[0]}")


def demo_polynomial_features():
    """PolynomialFeatures 多项式特征"""
    print("=" * 50)
    print("4. PolynomialFeatures")
    print("=" * 50)
    
    from sklearn.preprocessing import PolynomialFeatures
    
    X = np.array([[1, 2], [3, 4]])
    
    # degree=2: 生成 1, a, b, a², ab, b²
    poly2 = PolynomialFeatures(degree=2, include_bias=True)
    X_poly2 = poly2.fit_transform(X)
    print(f"degree=2:")
    print(f"  特征: {poly2.get_feature_names_out()}")
    print(f"  结果:\n{X_poly2}")
    
    # 只保留交互项
    poly_inter = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_inter = poly_inter.fit_transform(X)
    print(f"\ninteraction_only=True:")
    print(f"  特征: {poly_inter.get_feature_names_out()}")


def demo_variance_threshold():
    """VarianceThreshold 方差过滤"""
    print("=" * 50)
    print("5. VarianceThreshold 方差过滤")
    print("=" * 50)
    
    from sklearn.feature_selection import VarianceThreshold
    
    # 创建数据，第3列是常量
    X = np.array([
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1]
    ])
    
    var_thresh = VarianceThreshold(threshold=0)
    X_filtered = var_thresh.fit_transform(X)
    
    print(f"原始形状: {X.shape}")
    print(f"各特征方差: {var_thresh.variances_}")
    print(f"过滤后形状: {X_filtered.shape}")


def demo_select_k_best():
    """SelectKBest 特征选择"""
    print("=" * 50)
    print("6. SelectKBest 特征选择")
    print("=" * 50)
    
    from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    
    # 使用 F 检验
    selector = SelectKBest(score_func=f_classif, k=2)
    X_selected = selector.fit_transform(X, y)
    
    print(f"原始特征数: {X.shape[1]}")
    print(f"选择后特征数: {X_selected.shape[1]}")
    print(f"各特征得分: {selector.scores_.round(2)}")
    print(f"选中的特征: {np.array(iris.feature_names)[selector.get_support()]}")


def demo_rfe():
    """RFE 递归特征消除"""
    print("=" * 50)
    print("7. RFE 递归特征消除")
    print("=" * 50)
    
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    
    rfe = RFE(
        estimator=LogisticRegression(max_iter=1000),
        n_features_to_select=2,
        step=1  # 每次移除1个特征
    )
    X_rfe = rfe.fit_transform(X, y)
    
    print(f"特征排名: {rfe.ranking_}")
    print(f"选中特征: {np.array(iris.feature_names)[rfe.support_]}")


def demo_select_from_model():
    """SelectFromModel 基于模型的特征选择"""
    print("=" * 50)
    print("8. SelectFromModel")
    print("=" * 50)
    
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestClassifier
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    sfm = SelectFromModel(rf, threshold='median')
    X_sfm = sfm.fit_transform(X, y)
    
    print(f"特征重要性: {sfm.estimator_.feature_importances_.round(3)}")
    print(f"阈值: {sfm.threshold_:.3f}")
    print(f"选中特征: {np.array(iris.feature_names)[sfm.get_support()]}")


def demo_all():
    """运行所有演示"""
    demo_count_vectorizer()
    print()
    demo_tfidf_vectorizer()
    print()
    demo_dict_vectorizer()
    print()
    demo_polynomial_features()
    print()
    demo_variance_threshold()
    print()
    demo_select_k_best()
    print()
    demo_rfe()
    print()
    demo_select_from_model()


if __name__ == "__main__":
    demo_all()
