"""
交互式可视化
对应文档: ../../docs/visualization/09-interactive.md
"""

import numpy as np
import pandas as pd


def demo_plotly_basics():
    """演示 Plotly 基础"""
    print("=" * 50)
    print("1. Plotly 基础")
    print("=" * 50)
    
    print("Plotly 是交互式可视化库，支持:")
    print("  - 缩放、平移、悬停提示")
    print("  - 导出为 HTML")
    print("  - 在 Jupyter Notebook 中直接渲染")
    print()
    
    print("基本用法:")
    print("  import plotly.express as px")
    print("  import plotly.graph_objects as go")
    print()
    
    print("Plotly Express (快速绑图):")
    print("  fig = px.scatter(df, x='x', y='y', color='category')")
    print("  fig.show()")


def demo_interactive_chart():
    """演示交互式图表"""
    print("=" * 50)
    print("2. 交互式图表示例")
    print("=" * 50)
    
    print("折线图:")
    print("  fig = px.line(df, x='date', y='value', title='Time Series')")
    print()
    
    print("散点图:")
    print("  fig = px.scatter(df, x='x', y='y', size='size', color='group')")
    print()
    
    print("柱状图:")
    print("  fig = px.bar(df, x='category', y='value', color='group')")
    print()
    
    print("3D 散点图:")
    print("  fig = px.scatter_3d(df, x='x', y='y', z='z', color='label')")


def demo_plotly_tips():
    """演示 Plotly 技巧"""
    print("=" * 50)
    print("3. Plotly 实用技巧")
    print("=" * 50)
    
    print("保存为 HTML:")
    print("  fig.write_html('chart.html')")
    print()
    
    print("保存为图片:")
    print("  fig.write_image('chart.png')")
    print()
    
    print("自定义布局:")
    print("  fig.update_layout(")
    print("      title='Chart Title',")
    print("      xaxis_title='X Axis',")
    print("      yaxis_title='Y Axis',")
    print("      template='plotly_dark'")
    print("  )")


def demo_all():
    """运行所有演示"""
    demo_plotly_basics()
    print()
    demo_interactive_chart()
    print()
    demo_plotly_tips()


if __name__ == "__main__":
    demo_all()
