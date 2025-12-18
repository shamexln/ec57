import os
import subprocess
import pandas as pd
import glob

def run_rxr():
    # 1. 获取目录下所有的 .atr 文件
    atr_files = glob.glob("*.atr")
    if not atr_files:
        print("未找到 .atr 文件")
        return

    # 2. 清理旧的输出文件，以确保 rxr 重新开始累加
    for f in ["1.txt", "2.txt"]:
        if os.path.exists(f):
            os.remove(f)

    # 3. 遍历并运行 rxr 命令
    # 命令格式: rxr -r <record> -a atr bio -L 1.txt 2.txt
    for atr_file in atr_files:
        record = os.path.splitext(atr_file)[0]
        # 跳过某些特殊的或是重复的记录名（如果需要）
        # 这里直接根据文件名作为 record
        cmd = ["rxr.exe", "-r", record, "-a", "atr", "bio", "-L", "1.txt", "2.txt"]
        print(f"正在处理记录: {record} ...")
        try:
            # 使用 shell=True 在某些 Windows 环境下更稳，但这里直接传列表也可以
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"处理记录 {record} 时出错: {e}")

    # 4. 读取生成的 1.txt 并导出为 Excel
    if os.path.exists("1.txt"):
        try:
            # 1.txt 看起来是空格分隔的，且带有标题
            # 观察到标题行和数据行，使用 sep=r'\s+'
            df = pd.read_csv("1.txt", sep=r'\s+')
            
            output_excel = "report_rxr1.xlsx"
            df.to_excel(output_excel, index=False)
            print(f"报告已成功导出至: {output_excel}")
        except Exception as e:
            print(f"解析 1.txt 或导出 Excel 时出错: {e}")
            # 备选方案：保存为 CSV
            df.to_csv("report_rxr1.csv", index=False)
            print("已回退并保存为 report_rxr1.csv")
    else:
        print("未生成 1.txt，请检查 rxr.exe 是否运行正确。")

    if os.path.exists("2.txt"):
        try:
            # 1.txt 看起来是空格分隔的，且带有标题
            # 观察到标题行和数据行，使用 sep=r'\s+'
            df = pd.read_csv("2.txt", sep=r'\s+')

            output_excel = "report_rxr2.xlsx"
            df.to_excel(output_excel, index=False)
            print(f"报告已成功导出至: {output_excel}")
        except Exception as e:
            print(f"解析 2.txt 或导出 Excel 时出错: {e}")
            # 备选方案：保存为 CSV
            df.to_csv("report_rxr2.csv", index=False)
            print("已回退并保存为 report_rxr2.csv")
    else:
        print("未生成 2.txt，请检查 rxr.exe 是否运行正确。")

if __name__ == "__main__":
    run_rxr()
