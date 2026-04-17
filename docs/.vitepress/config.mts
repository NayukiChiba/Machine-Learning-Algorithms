import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

const customElements = [
  'mjx-container', 'mjx-assistive-mml', 'math', 'maction', 'maligngroup',
  'malignmark', 'menclose', 'merror', 'mfenced', 'mfrac', 'mi', 'mlongdiv',
  'mmultiscripts', 'mn', 'mo', 'mover', 'mpadded', 'mphantom', 'mroot',
  'mrow', 'ms', 'mscarries', 'mscarry', 'msgroup', 'mstack', 'msline',
  'mspace', 'msqrt', 'msrow', 'mstyle', 'msub', 'msup', 'msubsup',
  'mtable', 'mtd', 'mtext', 'mtr', 'munder', 'munderover', 'semantics',
  'annotation', 'annotation-xml',
]

function algoGroup(base: string, slug: string, label: string) {
  const p = `${base}/${slug}`
  return {
    text: label,
    collapsed: true,
    items: [
      { text: '总览', link: `${p}/` },
      { text: '01 数学原理', link: `${p}/01-mathematics` },
      { text: '02 数据构成', link: `${p}/02-data` },
      { text: '03 思路与直觉', link: `${p}/03-intuition` },
      { text: '04 模型构建', link: `${p}/04-model` },
      { text: '05 训练与预测', link: `${p}/05-training-and-prediction` },
      { text: '06 评估与诊断', link: `${p}/06-evaluation` },
      { text: '07 工程实现', link: `${p}/07-implementation` },
      { text: '08 练习与文献', link: `${p}/08-exercises-and-references` },
    ],
  }
}

const foundationsSidebar = [
  { text: '入门', items: [{ text: '库生态总览', link: '/foundations/overview' }] },
  {
    text: 'NumPy',
    collapsed: false,
    items: [
      { text: '01 基础', link: '/foundations/numpy/01-basics' },
      { text: '02 创建数组', link: '/foundations/numpy/02-creation' },
      { text: '03 属性与 dtype', link: '/foundations/numpy/03-attributes' },
      { text: '04 索引', link: '/foundations/numpy/04-indexing' },
      { text: '05 运算与统计', link: '/foundations/numpy/05-operations' },
      { text: '06 线性代数', link: '/foundations/numpy/06-linalg' },
      { text: '07 变形', link: '/foundations/numpy/07-reshape' },
      { text: '08 广播', link: '/foundations/numpy/08-broadcasting' },
      { text: '09 拼接与拆分', link: '/foundations/numpy/09-concat-split' },
      { text: '10 文件 IO', link: '/foundations/numpy/10-file-io' },
      { text: '11 实用函数', link: '/foundations/numpy/11-utilities' },
      { text: '12 练习', link: '/foundations/numpy/12-practice' },
    ],
  },
  {
    text: 'Pandas',
    collapsed: true,
    items: [
      { text: '01 基础', link: '/foundations/pandas/01-basics' },
      { text: '02 IO', link: '/foundations/pandas/02-io' },
      { text: '03 选取', link: '/foundations/pandas/03-selection' },
      { text: '04 清洗', link: '/foundations/pandas/04-cleaning' },
      { text: '05 分组', link: '/foundations/pandas/05-groupby' },
      { text: '06 合并', link: '/foundations/pandas/06-merge' },
      { text: '07 时间序列', link: '/foundations/pandas/07-timeseries' },
      { text: '08 可视化', link: '/foundations/pandas/08-visualization' },
      { text: '09 进阶', link: '/foundations/pandas/09-advanced' },
    ],
  },
  {
    text: 'SciPy',
    collapsed: true,
    items: [
      { text: '01 概览', link: '/foundations/scipy/01-basics' },
      { text: '02 统计', link: '/foundations/scipy/02-stats' },
      { text: '03 假设检验', link: '/foundations/scipy/03-hypothesis' },
      { text: '04 优化', link: '/foundations/scipy/04-optimize' },
      { text: '05 插值', link: '/foundations/scipy/05-interpolate' },
      { text: '06 积分', link: '/foundations/scipy/06-integrate' },
      { text: '07 线代', link: '/foundations/scipy/07-linalg' },
      { text: '08 信号', link: '/foundations/scipy/08-signal' },
      { text: '09 稀疏', link: '/foundations/scipy/09-sparse' },
      { text: '10 空间', link: '/foundations/scipy/10-spatial' },
    ],
  },
  {
    text: '可视化',
    collapsed: true,
    items: [
      { text: '01 Matplotlib 基础', link: '/foundations/visualization/01-matplotlib-basics' },
      { text: '02 图表', link: '/foundations/visualization/02-matplotlib-charts' },
      { text: '03 Seaborn', link: '/foundations/visualization/03-seaborn' },
      { text: '04 Pandas 图', link: '/foundations/visualization/04-pandas-viz' },
      { text: '05 EDA', link: '/foundations/visualization/05-eda' },
      { text: '06 预处理可视化', link: '/foundations/visualization/06-preprocessing-viz' },
      { text: '07 决策可视化', link: '/foundations/visualization/07-model-decision' },
      { text: '08 评估图', link: '/foundations/visualization/08-model-evaluation' },
      { text: '09 交互', link: '/foundations/visualization/09-interactive' },
      { text: '10 报告', link: '/foundations/visualization/10-reporting' },
    ],
  },
  {
    text: 'Scikit-learn',
    collapsed: true,
    items: [
      { text: '01 入门', link: '/foundations/sklearn/01-basics' },
      { text: '02 预处理', link: '/foundations/sklearn/02-preprocessing' },
      { text: '03 特征工程', link: '/foundations/sklearn/03-feature-engineering' },
      { text: '04 Pipeline', link: '/foundations/sklearn/04-pipeline' },
      { text: '05 模型选择', link: '/foundations/sklearn/05-model-selection' },
      { text: '06 指标', link: '/foundations/sklearn/06-metrics' },
      { text: '07 模型', link: '/foundations/sklearn/07-models' },
      { text: '08 技巧', link: '/foundations/sklearn/08-tips' },
    ],
  },
  {
    text: '附录',
    items: [
      { text: '符号与记号', link: '/appendix/notation' },
      { text: '术语表', link: '/appendix/glossary' },
    ],
  },
]

const appendixSidebar = [
  {
    text: '附录',
    items: [
      { text: '符号与记号', link: '/appendix/notation' },
      { text: '术语表', link: '/appendix/glossary' },
      { text: '基础库总览', link: '/foundations/overview' },
    ],
  },
]

export default defineConfig({
  base: '/Machine-Learning-Algorithms/',
  title: '机器学习算法原理',
  description: 'Machine Learning Algorithms — 数学原理与代码实现',
  lang: 'zh-CN',

  markdown: {
    config: (md) => {
      md.use(mathjax3)
    },
  },

  vue: {
    template: {
      compilerOptions: {
        isCustomElement: (tag) => customElements.includes(tag),
      },
    },
  },

  head: [
    ['link', { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap' }],
  ],

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '基础库', link: '/foundations/overview' },
      { text: '分类', link: '/classification/knn/' },
      { text: '回归', link: '/regression/linear_regression/' },
      { text: '集成', link: '/ensemble/bagging/' },
      { text: '聚类', link: '/clustering/kmeans/' },
      { text: '降维', link: '/dimensionality/pca/' },
      { text: '概率', link: '/probabilistic/em/' },
    ],

    sidebar: {
      '/foundations/': foundationsSidebar,
      '/appendix/': appendixSidebar,
      '/classification/': [
        {
          text: '分类算法',
          items: [
            algoGroup('/classification', 'knn', 'KNN'),
            algoGroup('/classification', 'logistic_regression', '逻辑回归'),
            algoGroup('/classification', 'svc', 'SVM'),
            algoGroup('/classification', 'decision_tree', '决策树'),
            algoGroup('/classification', 'naive_bayes', '朴素贝叶斯'),
          ],
        },
      ],
      '/regression/': [
        {
          text: '回归算法',
          items: [
            algoGroup('/regression', 'linear_regression', '线性回归'),
            algoGroup('/regression', 'regularization', '正则化回归'),
            algoGroup('/regression', 'svr', 'SVR'),
            algoGroup('/regression', 'decision_tree', '决策树回归'),
          ],
        },
      ],
      '/ensemble/': [
        {
          text: '集成学习',
          items: [
            algoGroup('/ensemble', 'bagging', 'Bagging / RF'),
            algoGroup('/ensemble', 'gbdt', 'GBDT'),
            algoGroup('/ensemble', 'xgboost', 'XGBoost'),
            algoGroup('/ensemble', 'lightgbm', 'LightGBM'),
          ],
        },
      ],
      '/clustering/': [
        {
          text: '聚类算法',
          items: [
            algoGroup('/clustering', 'kmeans', 'K-Means'),
            algoGroup('/clustering', 'dbscan', 'DBSCAN'),
          ],
        },
      ],
      '/dimensionality/': [
        {
          text: '降维算法',
          items: [
            algoGroup('/dimensionality', 'pca', 'PCA'),
            algoGroup('/dimensionality', 'lda', 'LDA'),
          ],
        },
      ],
      '/probabilistic/': [
        {
          text: '概率模型',
          items: [
            algoGroup('/probabilistic', 'em', 'EM / GMM'),
            algoGroup('/probabilistic', 'hmm', 'HMM'),
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/NayukiChiba/Machine-Learning-Algorithms' },
    ],

    outline: {
      level: [2, 3],
      label: '目录',
    },

    search: {
      provider: 'local',
    },
  },
})
