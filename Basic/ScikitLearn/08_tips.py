"""
Scikit-learn 实用技巧
对应文档: ../docs/08_tips.md

使用方式：
    from code.08_tips import *
    demo_clone()
    demo_class_weight()
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


def demo_clone():
    """clone() 克隆模型"""
    print("=" * 50)
    print("1. clone() 克隆模型")
    print("=" * 50)
    
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # 克隆：复制参数，不复制训练状态
    rf_clone = clone(rf)
    
    print(f"原模型已训练: {hasattr(rf, 'estimators_')}")
    print(f"克隆模型已训练: {hasattr(rf_clone, 'estimators_')}")
    print(f"参数相同: {rf.get_params()['n_estimators'] == rf_clone.get_params()['n_estimators']}")


def demo_get_set_params():
    """get_params() / set_params()"""
    print("=" * 50)
    print("2. get_params() / set_params()")
    print("=" * 50)
    
    from sklearn.ensemble import RandomForestClassifier
    
    rf = RandomForestClassifier(n_estimators=100)
    
    # 获取参数
    print(f"get_params()['n_estimators']: {rf.get_params()['n_estimators']}")
    print(f"get_params()['max_depth']: {rf.get_params()['max_depth']}")
    
    # 设置参数
    rf.set_params(n_estimators=50, max_depth=5)
    print(f"\n修改后:")
    print(f"  n_estimators: {rf.get_params()['n_estimators']}")
    print(f"  max_depth: {rf.get_params()['max_depth']}")


def demo_class_weight():
    """class_weight 处理类别不平衡"""
    print("=" * 50)
    print("3. class_weight 处理类别不平衡")
    print("=" * 50)
    
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    
    # 创建不平衡数据
    X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"类别分布: {np.bincount(y_train)}")
    
    # 无权重
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    print(f"\n无权重 - 少数类F1: {classification_report(y_test, clf.predict(X_test), output_dict=True)['1']['f1-score']:.3f}")
    
    # 使用 balanced
    clf_balanced = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf_balanced.fit(X_train, y_train)
    print(f"balanced - 少数类F1: {classification_report(y_test, clf_balanced.predict(X_test), output_dict=True)['1']['f1-score']:.3f}")


def demo_compute_class_weight():
    """compute_class_weight()"""
    print("=" * 50)
    print("4. compute_class_weight()")
    print("=" * 50)
    
    from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
    
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    
    # 计算类别权重
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    print(f"类别分布: {np.bincount(y)}")
    print(f"类别权重: {dict(zip(np.unique(y), class_weights.round(3)))}")
    
    # 计算样本权重
    sample_weights = compute_sample_weight('balanced', y)
    print(f"样本权重 (唯一值): {np.unique(sample_weights).round(3)}")


def demo_custom_transformer():
    """自定义转换器"""
    print("=" * 50)
    print("5. 自定义转换器")
    print("=" * 50)
    
    from sklearn.base import BaseEstimator, TransformerMixin
    
    class LogTransformer(BaseEstimator, TransformerMixin):
        """自定义对数转换器"""
        
        def __init__(self, offset=1):
            self.offset = offset
        
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            return np.log(X + self.offset)
    
    X = np.array([[1, 10], [100, 1000]])
    log_trans = LogTransformer(offset=1)
    
    print(f"原始: {X.tolist()}")
    print(f"变换后: {log_trans.fit_transform(X).round(3).tolist()}")


def demo_model_persistence():
    """模型持久化 (joblib)"""
    print("=" * 50)
    print("6. 模型持久化 (joblib)")
    print("=" * 50)
    
    import joblib
    import os
    from tempfile import mkdtemp
    from sklearn.ensemble import RandomForestClassifier
    
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    temp_dir = mkdtemp()
    
    # 保存
    path = os.path.join(temp_dir, 'model.joblib')
    joblib.dump(rf, path)
    print(f"保存大小: {os.path.getsize(path)/1024:.1f} KB")
    
    # 加载
    rf_loaded = joblib.load(path)
    print(f"加载后预测一致: {(rf_loaded.predict(X_test) == rf.predict(X_test)).all()}")
    
    # 压缩保存
    path_compressed = os.path.join(temp_dir, 'model_compressed.joblib')
    joblib.dump(rf, path_compressed, compress=3)
    print(f"压缩后大小: {os.path.getsize(path_compressed)/1024:.1f} KB")


def demo_sklearn_config():
    """sklearn 全局配置"""
    print("=" * 50)
    print("7. sklearn 全局配置")
    print("=" * 50)
    
    from sklearn import set_config, get_config
    
    print(f"当前配置: {get_config()}")
    
    # 设置输出为 pandas DataFrame
    set_config(transform_output='pandas')
    print("设置 transform_output='pandas'")
    
    # 恢复默认
    set_config(transform_output='default')
    print("恢复 transform_output='default'")


def demo_version_check():
    """版本检查"""
    print("=" * 50)
    print("8. 版本检查")
    print("=" * 50)
    
    import sklearn
    from packaging import version
    
    print(f"sklearn 版本: {sklearn.__version__}")
    
    if version.parse(sklearn.__version__) >= version.parse("1.0"):
        print("✓ 版本 >= 1.0")
    
    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        print("✓ 版本 >= 1.2, 支持 set_output API")


def demo_all_estimators():
    """查看所有可用估计器"""
    print("=" * 50)
    print("9. 查看可用估计器")
    print("=" * 50)
    
    from sklearn.utils import all_estimators
    
    classifiers = all_estimators(type_filter='classifier')
    regressors = all_estimators(type_filter='regressor')
    transformers = all_estimators(type_filter='transformer')
    
    print(f"分类器数量: {len(classifiers)}")
    print(f"回归器数量: {len(regressors)}")
    print(f"转换器数量: {len(transformers)}")
    print(f"分类器前5个: {[name for name, _ in classifiers[:5]]}")


def demo_all():
    """运行所有演示"""
    demo_clone()
    print()
    demo_get_set_params()
    print()
    demo_class_weight()
    print()
    demo_compute_class_weight()
    print()
    demo_custom_transformer()
    print()
    demo_model_persistence()
    print()
    demo_sklearn_config()
    print()
    demo_version_check()
    print()
    demo_all_estimators()


if __name__ == "__main__":
    demo_all()
