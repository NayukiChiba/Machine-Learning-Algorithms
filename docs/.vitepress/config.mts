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
      { text: '分类', link: '/classification/logistic_regression' },
      { text: '回归', link: '/regression/linear_regression' },
      { text: '集成', link: '/ensemble/bagging' },
      { text: '聚类', link: '/clustering/kmeans' },
      { text: '降维', link: '/dimensionality/pca' },
      { text: '概率', link: '/probabilistic/em' },
    ],

    sidebar: {
      '/classification/': [
        {
          text: '分类算法',
          items: [
            { text: 'KNN 近邻分类', link: '/classification/knn' },
            { text: '逻辑回归', link: '/classification/logistic_regression' },
            { text: 'SVM 支持向量机', link: '/classification/svc' },
            { text: '决策树', link: '/classification/decision_tree' },
            { text: '朴素贝叶斯', link: '/classification/naive_bayes' },
          ],
        },
      ],
      '/regression/': [
        {
          text: '回归算法',
          items: [
            { text: '线性回归', link: '/regression/linear_regression' },
            { text: '正则化回归', link: '/regression/regularization' },
            { text: 'SVR 支持向量回归', link: '/regression/svr' },
            { text: '决策树回归', link: '/regression/decision_tree' },
          ],
        },
      ],
      '/ensemble/': [
        {
          text: '集成学习',
          items: [
            { text: 'Bagging 与 Random Forest', link: '/ensemble/bagging' },
            { text: 'GBDT 梯度提升树', link: '/ensemble/gbdt' },
            { text: 'XGBoost', link: '/ensemble/xgboost' },
            { text: 'LightGBM', link: '/ensemble/lightgbm' },
          ],
        },
      ],
      '/clustering/': [
        {
          text: '聚类算法',
          items: [
            { text: 'K-Means', link: '/clustering/kmeans' },
            { text: 'DBSCAN', link: '/clustering/dbscan' },
          ],
        },
      ],
      '/dimensionality/': [
        {
          text: '降维算法',
          items: [
            { text: 'PCA 主成分分析', link: '/dimensionality/pca' },
            { text: 'LDA 线性判别分析', link: '/dimensionality/lda' },
          ],
        },
      ],
      '/probabilistic/': [
        {
          text: '概率模型',
          items: [
            { text: 'EM 算法与 GMM', link: '/probabilistic/em' },
            { text: 'HMM 隐马尔可夫模型', link: '/probabilistic/hmm' },
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
