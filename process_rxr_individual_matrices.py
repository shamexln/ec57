import os
import subprocess
import re
import csv
import glob
import shutil

def run_rxr_command(record, directory):
    """运行 rxr.exe 命令并捕获输出"""
    output_file = f"temp_{record}.txt"
    # rxr.exe 路径
    rxr_path = os.path.abspath("rxr.exe")
    
    # 确保 DLL 在目标目录
    if not os.path.exists(os.path.join(directory, "wfdb-10.7.dll")):
        if os.path.exists("wfdb-10.7.dll"):
            shutil.copy("wfdb-10.7.dll", os.path.join(directory, "wfdb-10.7.dll"))

    cmd = [rxr_path, "-r", record, "-a", "atr", "bio", "-s", output_file]
    try:
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
    
    matrices = []
    # 正则表达式匹配矩阵部分
    # 寻找包含 | 的行
    matrix_sections = re.findall(r'(\s+\d+\s+\|\s+.*?\n(?:.*?\n)*?\s+>5\s+\|\s+.*?\n)', content)
    
    for section in matrix_sections:
        matrix = []
        lines = section.strip().split('\n')
        for line in lines:
            if '|' not in line:
                continue
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

def save_matrix_to_csv(matrix, filename):
    """将矩阵保存为 CSV 文件"""
    header = ["Row/Col", "0", "1", "2", "3", "4", "5", ">5"]
    rows = ["0", "1", "2", "3", "4", "5", ">5"]
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, row in enumerate(matrix):
            writer.writerow([rows[i]] + row)

def main():
    data_dir = "run_out"
    output_dir = "matrix_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    atr_files = glob.glob(os.path.join(data_dir, "*.atr"))
    records = sorted(list(set([os.path.splitext(os.path.basename(f))[0] for f in atr_files])))
    
    for record in records:
        # 检查对应的 .bio 文件是否存在
        if not os.path.exists(os.path.join(data_dir, f"{record}.bio")):
            # print(f"Skipping {record}: .bio file not found")
            continue
            
        print(f"Processing record: {record}")
        output_file = run_rxr_command(record, data_dir)
        if output_file:
            S, P = parse_matrices(output_file)
            if S and P:
                s_filename = os.path.join(output_dir, f"{record}S.csv")
                p_filename = os.path.join(output_dir, f"{record}P.csv")
                
                save_matrix_to_csv(S, s_filename)
                save_matrix_to_csv(P, p_filename)
                print(f"  Saved {record}S.csv and {record}P.csv")
            else:
                print(f"  Failed to parse matrices for {record}")
            
            # 删除临时文件
            if os.path.exists(output_file):
                os.remove(output_file)
    
    print(f"All processing completed. Matrices are saved in '{output_dir}' directory.")

if __name__ == "__main__":
    main()
