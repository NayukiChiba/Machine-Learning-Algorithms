"""
数据导入与导出
对应文档: ../../docs/foundations/pandas/02-io.md

使用方式：
    python -m Basic.Pandas.02_io
"""

import pandas as pd

from . import output_path as get_output_path


def read_csv():
    """演示 CSV 文件读取"""
    print("=" * 50)
    print("1. CSV 文件读取")
    print("=" * 50)

    # 创建示例数据并保存
    df = pd.DataFrame(
        {
            "Name": ["Alice", "Bob", "Charlie"],
            "Age": [25, 30, 35],
            "Score": [85.5, 90.0, 78.5],
        }
    )

    # 直接保存到输出目录，方便后续查看
    output_file = get_output_path("pandas_demo.csv")
    df.to_csv(output_file, index=False)
    print(f"已保存示例数据到 {output_file}")

    # 读取 CSV
    df_read = pd.read_csv(output_file)
    print("\n读取的数据:")
    print(df_read)

    # 读取时指定参数
    print("\n常用读取参数:")
    print("  - sep: 分隔符 (默认',')")
    print("  - header: 表头行号")
    print("  - names: 自定义列名")
    print("  - usecols: 读取指定列")
    print("  - nrows: 读取行数")
    print("  - skiprows: 跳过行数")
    print("  - encoding: 编码格式")


def read_excel():
    """演示 Excel 文件读写（概念演示）"""
    print("=" * 50)
    print("2. Excel 文件读写")
    print("=" * 50)

    print("读取 Excel 文件:")
    print("  df = pd.read_excel('file.xlsx', sheet_name='Sheet1')")
    print()
    print("保存到 Excel:")
    print(
        "  df.to_excel('outputs/pandas/pandas_demo.xlsx', sheet_name='Sheet1', index=False)"
    )
    print()
    print("常用参数:")
    print("  - sheet_name: 工作表名或索引")
    print("  - header: 表头行号")
    print("  - usecols: 读取的列范围 (如 'A:C' 或 [0,1,2])")
    print("  - engine: 解析引擎 ('openpyxl', 'xlrd')")


def read_json():
    """演示 JSON 文件读写"""
    print("=" * 50)
    print("3. JSON 文件读写")
    print("=" * 50)

    # 创建示例数据
    df = pd.DataFrame(
        {
            "Name": ["Alice", "Bob"],
            "Age": [25, 30],
            "Skills": [["Python", "SQL"], ["Java", "C++"]],
        }
    )

    # 保存为 JSON
    output_file = get_output_path("pandas_demo.json")
    df.to_json(output_file, orient="records", indent=2, force_ascii=False)
    print(f"已保存到 {output_file}")

    # 读取 JSON
    df_read = pd.read_json(output_file)
    print("\n读取的数据:")
    print(df_read)

    print("\norient 参数选项:")
    print("  - 'records': [{列:值}, {列:值}]")
    print("  - 'columns': {列: {索引:值}}")
    print("  - 'index': {索引: {列:值}}")
    print("  - 'values': [[值, 值], [值, 值]]")


def export():
    """演示数据导出的各种格式"""
    print("=" * 50)
    print("4. 数据导出方法")
    print("=" * 50)

    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    csv_file = get_output_path("export_demo.csv")
    json_file = get_output_path("export_demo.json")
    html_file = get_output_path("export_demo.html")

    # 保存几种最常见的导出文件，便于直接查看结果
    df.to_csv(csv_file, index=False)
    df.to_json(json_file, orient="records", indent=2, force_ascii=False)
    df.to_html(html_file, index=False)

    print("导出方法:")
    print(f"  df.to_csv('{csv_file}')       # CSV")
    print("  df.to_excel('outputs/pandas/export_demo.xlsx')    # Excel")
    print(f"  df.to_json('{json_file}')     # JSON")
    print(f"  df.to_html('{html_file}')     # HTML")
    print("  df.to_sql('table', conn)    # SQL数据库")
    print("  df.to_pickle('file.pkl')    # Pickle序列化")
    print("  df.to_parquet('file.parquet') # Parquet")
    print()

    # 演示 to_string
    print("to_string() 输出:")
    print(df.to_string())


def run():
    """运行所有演示"""
    read_csv()
    print()
    read_excel()
    print()
    read_json()
    print()
    export()


if __name__ == "__main__":
    run()
