import os
import subprocess
import re
import csv
import glob

def run_rxr_command(record, directory):
    """运行 rxr.exe 命令并捕获输出"""
    output_file = f"temp_{record}.txt"
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
    """从文件中解析两个矩阵 (S矩阵 和 P矩阵)"""
    if not file_path or not os.path.exists(file_path):
        return None, None
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    matrices = []
    # 正则表达式匹配矩阵部分
    matrix_sections = re.findall(r'(\s+\d+\s+\|\s+.*?\n(?:.*?\n)*?\s+>5\s+\|\s+.*?\n)', content)
    
    for section in matrix_sections:
        matrix = []
        lines = section.strip().split('\n')
        for line in lines:
            parts = line.split('|')[1].strip().split()
            row = [int(x) for x in parts]
            if len(row) == 6:
                # 补上索引 0 的位置
                row = [0] + row
            matrix.append(row)
        if len(matrix) == 7:
            matrices.append(matrix)
            
    if len(matrices) >= 2:
        return matrices[0], matrices[1] # S 矩阵, P 矩阵
    return None, None

def calculate_metrics(S, P):
    """根据公式从 S 和 P 矩阵计算指标"""
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
    
    # 初始化总矩阵
    total_S = [[0 for _ in range(7)] for _ in range(7)]
    total_P = [[0 for _ in range(7)] for _ in range(7)]
    
    processed_count = 0
    for record in records:
        if not os.path.exists(os.path.join(data_dir, f"{record}.bio")):
            continue
            
        output_file = run_rxr_command(record, data_dir)
        if output_file:
            S, P = parse_matrices(output_file)
            if S and P:
                # 每一个 cell 累加
                for r in range(7):
                    for c in range(7):
                        total_S[r][c] += S[r][c]
                        total_P[r][c] += P[r][c]
                processed_count += 1
                print(f"Processed {record}")
            
            if os.path.exists(output_file):
                os.remove(output_file)
    
    print(f"Total records processed: {processed_count}")
    
    # 根据总矩阵计算指标
    metrics = calculate_metrics(total_S, total_P)
    
    # 写入 CSV
    # 我们不仅写入指标，还可以写入总矩阵的 cell 数据（如果需要）
    output_csv = 'rxr_total_matrix_results.csv'
    
    # 准备写入的数据，先放计算出的指标
    results = [metrics]
    
    keys = ["CTPs", "CFN", "CTPp", "CFP", "CSe", "C+P", "STPs", "SFN", "STPp", "SFP", "SSe", "S+P", "LTPs", "LFN", "LTPp", "LFP", "LSe", "L+P"]
    
    with open(output_csv, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    
    # 也可以把总矩阵保存下来供参考
    with open('total_S_matrix.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Row/Col", "0", "1", "2", "3", "4", "5", ">5"])
        for i, row in enumerate(total_S):
            label = str(i) if i < 6 else ">5"
            writer.writerow([label] + row)

    with open('total_P_matrix.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Row/Col", "0", "1", "2", "3", "4", "5", ">5"])
        for i, row in enumerate(total_P):
            label = str(i) if i < 6 else ">5"
            writer.writerow([label] + row)

    print(f"Done. Results saved to {output_csv}, total_S_matrix.csv and total_P_matrix.csv")

if __name__ == "__main__":
    main()
