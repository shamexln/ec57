import os
import subprocess
import re
import csv
import glob

def run_rxr_command(record, directory):
    """运行 rxr.exe 命令并捕获输出"""
    output_file = f"temp_{record}.txt"
    # 在指定目录下运行，或者确保 rxr.exe 能找到文件
    # 假设 rxr.exe 在当前目录，数据在 directory 目录
    # 我们需要切换到 directory 目录运行，或者让 rxr.exe 知道路径
    # rxr.exe 的 -r 参数通常需要 record 在当前目录或者 WFDB 路径下
    
    cmd = [os.path.abspath("rxr.exe"), "-r", record, "-a", "atr", "bio", "-s", output_file]
    try:
        # 复制必要的 DLL 到目标目录
        if not os.path.exists(os.path.join(directory, "wfdb-10.7.dll")):
            import shutil
            shutil.copy("wfdb-10.7.dll", os.path.join(directory, "wfdb-10.7.dll"))
        
        subprocess.run(cmd, cwd=directory, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return os.path.join(directory, output_file)
    except subprocess.CalledProcessError as e:
        print(f"Error processing record {record}: {e}")
        return None

def parse_matrices(file_path):
    """从文件中解析两个矩阵"""
    if not file_path or not os.path.exists(file_path):
        return None, None
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 定义矩阵解析逻辑
    # 矩阵通常在 "Algorithm Run Length" 之后，"_____" 之下
    # 我们需要找到两个矩阵
    
    matrices = []
    # 正则表达式匹配矩阵部分
    # 寻找包含 | 的行
    matrix_sections = re.findall(r'(\s+\d+\s+\|\s+.*?\n(?:.*?\n)*?\s+>5\s+\|\s+.*?\n)', content)
    
    for section in matrix_sections:
        matrix = []
        lines = section.strip().split('\n')
        for line in lines:
            # 提取数字部分
            # 格式类似: "  0 |         0    0    0    0    0    0" 或 "  >5 |    0    0    0    0    0    0    0"
            parts = line.split('|')[1].strip().split()
            # 填充前面的空缺（如果有的话，比如第一行可能少一个 0 对应的位置，但在 rxr 输出中通常是对应的）
            # 实际上根据 s1.txt，第一行是 "0 |         0    0    0    0    0    0" (6个值，对应 1, 2, 3, 4, 5, >5)
            # 其他行是 7 个值
            row = [int(x) for x in parts]
            if len(row) == 6:
                # 补上索引 0 的位置，通常是空的或者隐含为 0？
                # 仔细看 s1.txt:
                #           0    1    2    3    4    5   >5
                #      ____________________________________
                #    0 |         0    0    0    0    0    0
                # 第一行 0 对应的 Algorithm Run Length 0 是空的。
                row = [0] + row
            matrix.append(row)
        if len(matrix) == 7:
            matrices.append(matrix)
            
    if len(matrices) >= 2:
        return matrices[0], matrices[1] # S 矩阵, P 矩阵
    return None, None

def calculate_metrics(S, P):
    """根据公式计算指标"""
    # S[row][col], P[row][col]
    # row/col index 0-6 对应 0, 1, 2, 3, 4, 5, >5
    
    # CTPs = S22 + S23 + S24 + S25 + S26
    CTPs = S[2][2] + S[2][3] + S[2][4] + S[2][5] + S[2][6]
    # CFN = S20 + S21
    CFN = S[2][0] + S[2][1]
    # CTTp = P22 + P32 + P42 + P52 + P62
    CTPp = P[2][2] + P[3][2] + P[4][2] + P[5][2] + P[6][2]
    # CFP = P02 + P12
    CFP = P[0][2] + P[1][2]
    
    CSe = CTPs / (CTPs + CFN) if (CTPs + CFN) > 0 else 0
    C_plus_P = CTPp / (CTPp + CFP) if (CTPp + CFP) > 0 else 0
    
    # STPs = S33 + S34 + S35 + S36 + S43 + S44 + S45 + S46 + S53 + S54 + S55 + S56
    STPs = (S[3][3] + S[3][4] + S[3][5] + S[3][6] +
            S[4][3] + S[4][4] + S[4][5] + S[4][6] +
            S[5][3] + S[5][4] + S[5][5] + S[5][6])
    
    # SFN = S30 + S31 + S32 + S40 + S41 + S42 + S50 + S51 + S52
    SFN = (S[3][0] + S[3][1] + S[3][2] +
           S[4][0] + S[4][1] + S[4][2] +
           S[5][0] + S[5][1] + S[5][2])

    # STPp = P33 + P43 + P53 + P63 + P34 + P44 + P54 + P64 + P35 + P45 + P55 + P65
    STPp = (P[3][3] + P[4][3] + P[5][3] + P[6][3] +
            P[3][4] + P[4][4] + P[5][4] + P[6][4] +
            P[3][5] + P[4][5] + P[5][5] + P[6][5])
    
    # SFP = P03 + P13 + P23 + P04 + P14 + P24 + P05 + P15 + P25
    SFP = (P[0][3] + P[1][3] + P[2][3] +
           P[0][4] + P[1][4] + P[2][4] +
           P[0][5] + P[1][5] + P[2][5])
    
    SSe = STPs / (STPs + SFN) if (STPs + SFN) > 0 else 0
    S_plus_P = STPp / (STPp + SFP) if (STPp + SFP) > 0 else 0
    
    # LTPs = S66
    LTPs = S[6][6]
    # LFN = S60 + S61 + S62 + S63 + S64 + S65
    LFN = S[6][0] + S[6][1] + S[6][2] + S[6][3] + S[6][4] + S[6][5]
    
    # LTPp = P66
    LTPp = P[6][6]
    # LFP = P06 + P16 + P26 + P36 + P46 + P56
    LFP = P[0][6] + P[1][6] + P[2][6] + P[3][6] + P[4][6] + P[5][6]
    
    LSe = LTPs / (LTPs + LFN) if (LTPs + LFN) > 0 else 0
    L_plus_P = LTPp / (LTPp + LFP) if (LTPp + LFP) > 0 else 0
    
    return {
        "CTPs": CTPs, "CFN": CFN, "CTPp": CTPp, "CFP": CFP, "CSe": CSe, "C+P": C_plus_P,
        "STPs": STPs, "SFN": SFN, "STPp": STPp, "SFP": SFP, "SSe": SSe, "S+P": S_plus_P,
        "LTPs": LTPs, "LFN": LFN, "LTPp": LTPp, "LFP": LFP, "LSe": LSe, "L+P": L_plus_P
    }

def main():
    data_dir = "run_out"
    atr_files = glob.glob(os.path.join(data_dir, "*.atr"))
    records = sorted(list(set([os.path.splitext(os.path.basename(f))[0] for f in atr_files])))
    
    total_metrics = {
        "CTPs": 0, "CFN": 0, "CTPp": 0, "CFP": 0,
        "STPs": 0, "SFN": 0, "STPp": 0, "SFP": 0,
        "LTPs": 0, "LFN": 0, "LTPp": 0, "LFP": 0
    }
    
    results = []
    
    for record in records:
        print(f"Processing record: {record}")
        # 检查对应的 .bio 文件是否存在
        if not os.path.exists(os.path.join(data_dir, f"{record}.bio")):
            print(f"  Skipping {record}: .bio file not found")
            continue
            
        output_file = run_rxr_command(record, data_dir)
        if output_file:
            S, P = parse_matrices(output_file)
            if S and P:
                metrics = calculate_metrics(S, P)
                metrics['Record'] = record
                results.append(metrics)
                
                # 累加原始数值
                for key in total_metrics.keys():
                    total_metrics[key] += metrics[key]
            else:
                print(f"  Failed to parse matrices for {record}")
            
            # 删除临时文件
            if os.path.exists(output_file):
                os.remove(output_file)
    
    # 计算总指标
    final_CSe = total_metrics["CTPs"] / (total_metrics["CTPs"] + total_metrics["CFN"]) if (total_metrics["CTPs"] + total_metrics["CFN"]) > 0 else 0
    final_C_plus_P = total_metrics["CTPp"] / (total_metrics["CTPp"] + total_metrics["CFP"]) if (total_metrics["CTPp"] + total_metrics["CFP"]) > 0 else 0

    final_SSe = total_metrics["STPs"] / (total_metrics["STPs"] + total_metrics["SFN"]) if (total_metrics["STPs"] + total_metrics["SFN"]) > 0 else 0
    final_S_plus_P = total_metrics["STPp"] / (total_metrics["STPp"] + total_metrics["SFP"]) if (total_metrics["STPp"] + total_metrics["SFP"]) > 0 else 0

    final_LSe = total_metrics["LTPs"] / (total_metrics["LTPs"] + total_metrics["LFN"]) if (total_metrics["LTPs"] + total_metrics["LFN"]) > 0 else 0
    final_L_plus_P = total_metrics["LTPp"] / (total_metrics["LTPp"] + total_metrics["LFP"]) if (total_metrics["LTPp"] + total_metrics["LFP"]) > 0 else 0

    final_row = {
        "Record": "TOTAL",
        "CTPs": total_metrics["CTPs"], "CFN": total_metrics["CFN"], "CTPp": total_metrics["CTPp"], "CFP": total_metrics["CFP"],
        "CSe": final_CSe, "C+P": final_C_plus_P,
        "STPs": total_metrics["STPs"], "SFN": total_metrics["SFN"], "STPp": total_metrics["STPp"], "SFP": total_metrics["SFP"],
        "SSe": final_SSe, "S+P": final_S_plus_P,
        "LTPs": total_metrics["LTPs"], "LFN": total_metrics["LFN"], "LTPp": total_metrics["LTPp"], "LFP": total_metrics["LFP"],
        "LSe": final_LSe, "L+P": final_L_plus_P
    }
    results.append(final_row)
    
    # 写入 CSV
    keys = ["Record", "CTPs", "CFN", "CTPp", "CFP", "CSe", "C+P", "STPs", "SFN", "STPp", "SFP", "SSe", "S+P", "LTPs", "LFN", "LTPp", "LFP", "LSe", "L+P"]
    with open('rxr_metrics_summary.csv', 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    
    print("Done. Results saved to rxr_metrics_summary.csv")

if __name__ == "__main__":
    main()
