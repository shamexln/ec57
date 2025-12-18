"""
wfdb_runner.py

用途：
- 从本地 MIT/PhysioNet WFDB 数据目录读取指定记录（如 100、101）。
- 将可用导联重排为标准 12 导联顺序，生成 ECGAlg.exe 所需的 12 导联打包文件 <record>_trans.dat（int16，小端）。
- 自动生成 ECGAlg.exe 所需的 settings.ini，并调用 ECGAlg.exe 完成算法测试。
- 可选：把算法输出的 CSV（<record>.csv）转换为 WFDB 注释（<record>.bio）。

注意：
- 需要已安装 Python 包 wfdb、numpy、pandas（仅当启用 --to-ann 时需要 pandas）。
- 需要本地存在 ECGAlg.exe（默认当前目录），或通过 --exe 指定。
- MIT 数据目录示例：C:\\project\\mit-database，其中包含 100.dat/100.hea/100.atr 等文件。

示例：
python wfdb_runner.py --db "C:\\project\\mit-database" --record 100 --exe .\\ECGAlg.exe --to-ann
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np
import wfdb

from transform_data import process_record_to_trans_dat


# 标准 12 导联顺序
TARGET_12_LEADS: List[str] = [
    'I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
]


def read_record(db_dir: Path, record: str) -> Tuple[np.ndarray, List[str], float, list]:
    """读取 WFDB 记录。

    返回：
    - orig_sig: (channels, samples) float32（原始计数，不做单位换算）
    - orig_names: List[str]，大写
    - fs: 采样率
    - adc_gain: 每通道增益（来自头信息，可能为空）
    """
    rec_path = str((db_dir / record).as_posix())  # wfdb 接受 POSIX 风格分隔符
    sig, hdr = wfdb.rdsamp(rec_path)  # sig: (samples, channels)
    header = wfdb.rdheader(rec_path)

    fs = float(hdr['fs'])
    orig_sig = sig.T.astype(np.float32)
    orig_names = [str(n).upper() for n in hdr['sig_name']]
    adc_gain = header.adc_gain if header and hasattr(header, 'adc_gain') else []
    return orig_sig, orig_names, fs, adc_gain


def to_twelve_lead(orig_sig: np.ndarray, orig_names: List[str]) -> Tuple[np.ndarray, int, int]:
    """将原始导联映射/填充为 12 导联矩阵（12, samples）。

    返回 (twelve_lead, main_idx, second_idx)，其中 main/second 为算法 settings.ini 中的 0 基索引（MainLead/SecondLead）。
    优先选择：Main=II 或 MLII，其次 I；Second=V1（若无则 V2）。
    """
    name2idx = {n: i for i, n in enumerate(orig_names)}

    # 兼容常见别名：MLII 作为 II
    names_norm = set(orig_names)
    alias_to_real = {}
    if 'MLII' in names_norm and 'II' not in names_norm:
        alias_to_real['II'] = 'MLII'

    n_samples = orig_sig.shape[1]
    out = np.zeros((12, n_samples), dtype=np.float32)

    for t_idx, lead in enumerate(TARGET_12_LEADS):
        src_name = alias_to_real.get(lead, lead)
        if src_name in name2idx:
            out[t_idx, :] = orig_sig[name2idx[src_name]]

    # 选择 Main/Second 对应的 12 导联位置（0-based）
    def lead_pos_0based(lead_name: str) -> int | None:
        try:
            return TARGET_12_LEADS.index(lead_name)
        except ValueError:
            return None

    # Main: II/MLII > I > 任意第一个非零列
    main_pos = None
    if ('II' in names_norm) or ('MLII' in names_norm):
        main_pos = lead_pos_0based('II')
    if main_pos is None and 'I' in names_norm:
        main_pos = lead_pos_0based('I')
    if main_pos is None:
        # 找到第一个被填充的导联
        nz = [i for i in range(12) if np.any(out[i, :])]
        main_pos = (nz[0]) if nz else 0

    # Second: V1 > V2 > 与 Main 不同的下一个非零
    second_pos = None
    for cand in ['V1', 'V2']:
        if cand in names_norm:
            second_pos = lead_pos_0based(cand)
            break
    if second_pos is None or second_pos == main_pos:
        nz = [i for i in range(12) if np.any(out[i, :]) and i != main_pos]
        second_pos = nz[0] if nz else (10 if main_pos != 10 else 0)  # 默认 10=V4（0 基）

    return out, int(main_pos), int(second_pos)


def write_trans_dat(sig_chx: np.ndarray, out_path: Path) -> None:
    """写出 <record>_trans.dat，以“每个样本的 N 导联”顺序展平并写入文件（int16，小端）。
    输入数组形状为 (channels, samples)。逻辑参考 transform_data.py。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 转置为 (样本数, channels)，再展平为一维 [samples * channels]
    out_data = sig_chx.T.flatten()
    # 转为 int16 写出
    out_data.astype(np.int16).tofile(out_path)


def prepare_and_write_trans_dat(twelve: np.ndarray, main_pos: int, second_pos: int, leadnum: int, record: str, out_dir: Path) -> Path:
    """
    根据 leadnum 选择导联，进行 AD 转换并写出 trans.dat 文件。
    逻辑参考 transform_data.py。
    """
    # 1) 准备输出数据：按 leadnum 选择仅 2 导联或全部 12 导联
    if leadnum == 2:
        data_to_write = np.vstack([twelve[main_pos - 1, :], twelve[second_pos - 1, :]])
    else:
        data_to_write = twelve

    # 将物理单位（约 mV）转换为 AD 计数：AD = mV × ADperMV（官方示例为 200）
    AD_PER_MV = 200.0
    data_to_write = np.rint(data_to_write * AD_PER_MV)
    # 裁剪到 int16 范围，避免溢出
    data_to_write = np.clip(data_to_write, -32768, 32767)

    # 写出 trans.dat（按样本×通道交错）。
    trans_dat = out_dir / f"{record}_trans.dat"
    write_trans_dat(data_to_write, trans_dat)
    print(f"✅ 生成 {trans_dat}")
    return trans_dat

def write_settings(ini_path: Path, input_dat: Path, output_csv: Path, fs: float, main_lead: int, second_lead: int, lead_num: int) -> None:
    """生成 ECGAlg.exe 所需 settings.ini。"""
    # 按官方模板固定输出键顺序与默认值
    lines = [
        f"InputDataPath={input_dat}",
        f"OutputDataPath={output_csv}",
        "ADorCh=1",           # Adult/Child flag per official doc
        "PacePatient=0",
        f"Fs={int(fs)}",
        "ADperMV=200",
        "LeadType=2",
        "Norch=1",
        # 官方样例固定 LeadNum=2、MainLead=1、SecondLead=10（均为 0 基还是 1 基？官方文档为 0-11。
        # 按你的最新要求，严格写入如下三项：
        "LeadNum=2",
        "MainLead=1",
        "SecondLead=10",
    ]
    ini_path.write_text("\n".join(lines), encoding='utf-8')


def run_ecg_alg(exe_path: Path, work_dir: Path) -> None:
    """调用 ECGAlg.exe（在 work_dir 下读取 settings.ini）。

    将标准输出与标准错误重定向到 work_dir/ecg_log.txt，便于诊断。
    """
    if not exe_path.exists():
        raise FileNotFoundError(f"找不到 ECGAlg.exe: {exe_path}")
    log_path = work_dir / 'ecg_log.txt'
    with open(log_path, 'w', encoding='utf-8', errors='replace') as logf:
        subprocess.run([str(exe_path)], cwd=work_dir, check=True, stdout=logf, stderr=logf)


def csv_to_wfdb_ann(record: str, csv_path: Path, fs: float) -> None:
    """将算法输出的 <record>.csv 转为 WFDB 注释 <record>.bio（兼容两种列格式）。
    支持：
    - 有表头：IndexPos,time,beat type,heart rate；若存在 IndexPos 优先作为样本索引。
    """
    import pandas as pd

    def _normalize(cols):
        return [str(c).strip().lower().replace(' ', '') for c in cols]

    df = None
    try:
        df0 = pd.read_csv(csv_path)
        cols = _normalize(list(df0.columns))
        df0.columns = cols
        df = df0
    except Exception:
        df = None


    if 'indexpos' in df.columns:
        idx = pd.to_numeric(df['indexpos'], errors='coerce').to_numpy()
        samples = np.rint(idx).astype(np.int64)

    beat_col = 'beattype'

    beats_raw = df[beat_col].astype(str).tolist()
    # 符号映射
    # map_table = {'N': 'N', 'V': 'V', 'F': 'F', 'Q': 'Q', 'L': 'N'}
    # symbols = [map_table.get(b, 'N') for b in beats_raw]
    symbols =  beats_raw
    # 排序写出
    order = np.argsort(samples)
    samples = samples[order]
    symbols = [symbols[i] for i in order]
    wfdb.wrann(record, 'bio', sample=samples.tolist(), symbol=symbols)


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='使用 WFDB 读取记录并调用 ECGAlg.exe 进行测试')
    p.add_argument('--db', required=True, help=r'本地 WFDB 数据目录，例如 C\\\project\\mit-database')
    p.add_argument('--record', required=True, help='记录号（如 100、101）。不含扩展名')
    p.add_argument('--exe', default='./ECGAlg.exe', help='ECGAlg.exe 路径，默认当前目录')
    p.add_argument('--workdir', default='.', help='运行 ECGAlg.exe 的工作目录（settings.ini 与输出将放在此处）')
    p.add_argument('--outdir', default='.', help='生成 *_trans.dat 的输出目录（默认当前目录）')
    p.add_argument('--to-ann', action='store_true', help='将算法输出 CSV 转为 WFDB 注释 .bio')
    p.add_argument('--leadnum', type=int, choices=[2, 12], default=12, help='输出到 InputDataPath 的通道数：2=仅主/辅导联；12=全部12导联（推荐）')
    p.add_argument('--debug-dump', action='store_true', help='调试：导出首 1 秒 12 导联预览与十六进制转储，并记录 ECGAlg.exe 日志')
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    db_dir = Path(args.db)
    record = str(args.record)
    work_dir = Path(args.workdir)
    out_dir = Path(args.outdir)
    exe_path = Path(args.exe)

    if not db_dir.exists():
        print(f"错误：数据库目录不存在：{db_dir}")
        return 2

    # 1) 读取与重排
    orig_sig, orig_names, fs, _ = read_record(db_dir, record)
    twelve, main_pos, second_pos = to_twelve_lead(orig_sig, orig_names)

    # 2) 准备输出数据并写出 trans.dat
    record_path = str((db_dir / record).as_posix())
    #trans_dat = process_record_to_trans_dat(record_path, out_dir)
    trans_dat = prepare_and_write_trans_dat(twelve, main_pos, second_pos, args.leadnum, record, out_dir)



    # 调试转储：首 1 秒 12 导联（缩放后 int16）与前 64 字节十六进制
    if args.debug_dump:
        try:
            import csv
            preview_csv = out_dir / f"{record}_preview.csv"
            hex_txt = out_dir / f"{record}_hex.txt"
            # 为了调试转储，我们需要获取缩放后的数据
            # 这里简单复用逻辑或从文件读取，但为了保持 main 简洁且调试功能完整，我们重新获取一次 scaled_full
            if args.leadnum == 12:
                # 重新计算一次以便调试（或者修改 prepare_and_write_trans_dat 返回数据，但这里保持原样）
                scaled_full = np.clip(np.rint(twelve * 200.0), -32768, 32767)
            else:
                scaled_full = np.zeros((12, twelve.shape[1]), dtype=np.int16)
                m_val = np.clip(np.rint(twelve[main_pos - 1, :] * 200.0), -32768, 32767)
                s_val = np.clip(np.rint(twelve[second_pos - 1, :] * 200.0), -32768, 32767)
                scaled_full[1, :] = m_val  # II 位置
                scaled_full[10, :] = s_val # V5 位置
            sec = int(fs)
            with open(preview_csv, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(TARGET_12_LEADS)
                for i in range(min(sec, scaled_full.shape[1])):
                    w.writerow([int(scaled_full[ch, i]) for ch in range(12)])
            # 读取刚写出的二进制文件前 64 字节
            with open(trans_dat, 'rb') as bf:
                first = bf.read(64)
            hex_str = ' '.join(f'{b:02x}' for b in first)
            hex_txt.write_text(hex_str, encoding='utf-8')
            print(f"📝 调试文件：{preview_csv}, {hex_txt}")
        except Exception as e:
            print(f"⚠ 调试转储失败：{e}")

    # 3) 写 settings.ini（放在 work_dir，供 ECGAlg.exe 读取）
    ini_path = work_dir / 'settings.ini'
    output_csv = work_dir / f"{record}.csv"
    # ECGAlg.exe 对相对路径友好，尽量写相对路径
    input_for_ini = os.path.relpath(trans_dat, start=work_dir)
    output_for_ini = os.path.relpath(output_csv, start=work_dir)
    write_settings(ini_path, Path(input_for_ini), Path(output_for_ini), fs, main_pos, second_pos, args.leadnum)
    print(f"✅ 生成 {ini_path}")

    # 4) 运行 ECGAlg.exe
    print("▶ 运行 ECGAlg.exe ...")
    run_ecg_alg(exe_path, work_dir)
    print(f"✅ ECGAlg.exe 完成，输出：{output_csv}")

    # 5) 可选：转换为 WFDB 注释
    if args.to_ann:
        try:
            csv_to_wfdb_ann(record, output_csv, fs)
            print(f"✅ 已写注释 {record}.bio")
        except Exception as e:
            print(f"⚠ 转换注释失败：{e}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
