"""
Scikit-learn Pipeline 流水线
对应文档: ../docs/04_pipeline.md

使用方式：
    from code.04_pipeline import *
    demo_basic_pipeline()
    demo_column_transformer()
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split


def demo_basic_pipeline():
    """Pipeline 基础用法"""
    print("=" * 50)
    print("1. Pipeline 基础")
    print("=" * 50)

    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 方式1: Pipeline（显式命名）
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("pca", PCA(n_components=2)), ("svm", SVC())]
    )

    pipe.fit(X_train, y_train)
    print(f"Pipeline 准确率: {pipe.score(X_test, y_test):.4f}")
    print(f"步骤名称: {[name for name, _ in pipe.steps]}")

    # 方式2: make_pipeline（自动命名）
    pipe_auto = make_pipeline(StandardScaler(), PCA(n_components=2), SVC())
    print(f"自动命名: {[name for name, _ in pipe_auto.steps]}")


def demo_access_steps():
    """访问 Pipeline 步骤"""
    print("=" * 50)
    print("2. 访问 Pipeline 步骤")
    print("=" * 50)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC

    pipe = Pipeline(
        [("scaler", StandardScaler()), ("pca", PCA(n_components=2)), ("svm", SVC())]
    )

    # 访问方式
    print(f"pipe.steps: {pipe.steps}")
    print(f"pipe.named_steps['pca']: {pipe.named_steps['pca']}")
    print(f"pipe[0]: {pipe[0]}")
    print(f"pipe[-1]: {pipe[-1]}")


def demo_set_params():
    """Pipeline 参数设置"""
    print("=" * 50)
    print("3. Pipeline 参数设置")
    print("=" * 50)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2)),
            ("svm", SVC(C=1.0)),
        ]
    )

    print(
        f"修改前: PCA n_components={pipe.named_steps['pca'].n_components}, SVM C={pipe.named_steps['svm'].C}"
    )

    # 格式: 步骤名__参数名
    pipe.set_params(pca__n_components=3, svm__C=10)

    print(
        f"修改后: PCA n_components={pipe.named_steps['pca'].n_components}, SVM C={pipe.named_steps['svm'].C}"
    )


def demo_pipeline_gridsearch():
    """Pipeline + GridSearchCV"""
    print("=" * 50)
    print("4. Pipeline + GridSearchCV")
    print("=" * 50)

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    pipe = make_pipeline(StandardScaler(), SVC())

    param_grid = {"svc__C": [0.1, 1, 10], "svc__kernel": ["linear", "rbf"]}

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy")
    grid.fit(X, y)

    print(f"最佳参数: {grid.best_params_}")
    print(f"最佳得分: {grid.best_score_:.4f}")


def demo_skip_step():
    """跳过 Pipeline 步骤"""
    print("=" * 50)
    print("5. 跳过 Pipeline 步骤")
    print("=" * 50)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    pipe = Pipeline(
        [("scaler", StandardScaler()), ("pca", PCA(n_components=2)), ("svm", SVC())]
    )

    # 设为 'passthrough' 跳过该步骤
    pipe.set_params(pca="passthrough")
    pipe.fit(X_train, y_train)

    print(f"跳过 PCA 后准确率: {pipe.score(X_test, y_test):.4f}")
    print(f"当前 pca 步骤: {pipe.named_steps['pca']}")


def demo_column_transformer():
    """ColumnTransformer 混合类型处理"""
    print("=" * 50)
    print("6. ColumnTransformer 混合类型处理")
    print("=" * 50)

    from sklearn.compose import ColumnTransformer, make_column_selector as selector
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    # 创建混合类型数据
    df = pd.DataFrame(
        {
            "年龄": [25, 30, np.nan, 40],
            "收入": [50000, 60000, 55000, np.nan],
            "学历": ["本科", "硕士", "本科", "博士"],
        }
    )
    y = [0, 1, 0, 1]

    print(f"原始数据:\n{df}")

    # 完整流水线
    full_pipe = Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        (
                            "num",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            selector(dtype_include="number"),
                        ),
                        (
                            "cat",
                            Pipeline(
                                [
                                    (
                                        "imputer",
                                        SimpleImputer(strategy="most_frequent"),
                                    ),
                                    (
                                        "onehot",
                                        OneHotEncoder(
                                            handle_unknown="ignore", sparse_output=False
                                        ),
                                    ),
                                ]
                            ),
                            selector(dtype_include="object"),
                        ),
                    ]
                ),
            ),
            ("classifier", LogisticRegression()),
        ]
    )

    full_pipe.fit(df, y)
    print(f"\n特征名: {full_pipe.named_steps['preprocessor'].get_feature_names_out()}")


def demo_transformed_target():
    """TransformedTargetRegressor 目标变量变换"""
    print("=" * 50)
    print("7. TransformedTargetRegressor")
    print("=" * 50)

    from sklearn.compose import TransformedTargetRegressor
    from sklearn.linear_model import LinearRegression

    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 普通回归
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print(f"普通回归 R²: {lr.score(X_test, y_test):.4f}")

    # 对数变换目标变量
    ttr = TransformedTargetRegressor(
        regressor=LinearRegression(),
        func=np.log1p,  # y -> log(1+y)
        inverse_func=np.expm1,  # 逆变换
    )
    ttr.fit(X_train, y_train)
    print(f"目标变量对数变换后 R²: {ttr.score(X_test, y_test):.4f}")


def demo_all():
    """运行所有演示"""
    demo_basic_pipeline()
    print()
    demo_access_steps()
    print()
    demo_set_params()
    print()
    demo_pipeline_gridsearch()
    print()
    demo_skip_step()
    print()
    demo_column_transformer()
    print()
    demo_transformed_target()


if __name__ == "__main__":
    demo_all()
