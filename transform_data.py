"""
本脚本用途：
- 从 MIT-BIH 心律失常数据库读取编号为 100 的记录（需要本地存在 100.dat/100.hea 等文件，或通过 WFDB 的默认搜索路径可以访问）。
- 将记录中的原始导联重排为目标的 12 导联顺序，并以每个采样点包含 12 个导联值的顺序展开后，保存为二进制文件 100_trans.DAT（int16）。

处理流程概述：
1) 读取波形与头文件，拿到采样频率、导联名称、增益等元信息。
2) 将原信号转为 float32 并按增益进行标度（把原单位计数转换为物理量刻度）。
3) 构造到目标 12 导联顺序的映射，把存在的导联复制到相应位置；缺失导联保持为 0。
4) 以 [样本1的12导联, 样本2的12导联, ...] 的顺序一维化并写入 .DAT 文件（小端，int16）。

注意事项：
- WFDB 数据集中，常见导联名大小写不一，脚本统一转为大写后进行匹配。
- 如果原记录只包含少数导联（如 MLII 和 V1），其他导联会是 0 填充。
- 若需要处理其他记录号，请把 './100' 改为相应路径前缀。
"""

import wfdb
import numpy as np
from pathlib import Path
from typing import  Tuple


def process_record_to_trans_dat(record_path: str, out_dir: Path) -> Tuple[Path, int, int]:
    """
    读取 WFDB 记录，按 transform_data.py 原始算法逻辑处理并写出 trans.dat。
    包含 adc_gain 标度、12 导联重排（含 MLII）。
    """
    record_name = Path(record_path).name
    # 1) 读取信号与头信息
    sig, hdr = wfdb.rdsamp(record_path)
    header = wfdb.rdheader(record_path)

    # 读取增益和采样率（供参考）
    adc_gain = header.adc_gain             # 每个通道的增益（将原始计数转物理单位的比例）
    fs = hdr['fs']                         # 采样率（例如 360 Hz）

    # 将信号转为 [channels, samples]，并转为 float32 以便数值处理
    orig_sig = sig.T.astype(np.float32)    # 形状: (实际导联数, 样本数)

    # 按通道增益做标度（若某些通道不存在相应增益索引，请做边界检查）
    if orig_sig.shape[0] > 0 and len(adc_gain) > 0:
        orig_sig[0, :] = orig_sig[0, :] * adc_gain[0]
    if orig_sig.shape[0] > 1 and len(adc_gain) > 1:
        orig_sig[1, :] = orig_sig[1, :] * adc_gain[1]

    # 统一大写导联名，便于匹配
    orig_names = [c.upper() for c in hdr['sig_name']]

    # ------------------------------------------------
    # 2) 定义目标的 12 导联顺序
    # ------------------------------------------------
    # 常见 12 导联顺序：I, II(此处对应 MLII), III, aVR, aVL, aVF, V1-V6
    # 若原记录没有全部导联，缺失项将保持为 0
    target_order = ['I', 'MLII', 'III',
                    'AVR', 'AVL', 'AVF',
                    'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # 建立 原名称 -> 索引 的映射
    name2idx = {name: idx for idx, name in enumerate(orig_names)}

    # 输出原记录的前两条导联在目标顺序中的位置（若存在）
    try:
        main_idx = target_order.index(orig_names[0]) if len(orig_names) > 0 else 'N/A'
        second_idx = target_order.index(orig_names[1]) if len(orig_names) > 1 else 'N/A'
        print("MainLead: {}, SecondLead: {}".format(main_idx, second_idx))
    except ValueError:
        # 若找不到对应导联名则忽略
        print("MainLead/SecondLead 不在目标 12 导联列表中")

    # 预分配 12 导联矩阵（12, 样本数）
    n_samples = orig_sig.shape[1]
    twelve_lead = np.zeros((12, n_samples), dtype=np.float32)

    # 将存在的导联复制到对应位置；缺失的保持为 0
    for tgt_idx, tgt_name in enumerate(target_order):
        if tgt_name in name2idx:
            twelve_lead[tgt_idx, :] = orig_sig[name2idx[tgt_name]]

    # ------------------------------------------------
    # 3) 以 “每个样本 of 12 导联” 顺序展平并写入文件
    # ------------------------------------------------
    # 转置为 (样本数, 12)，再展平为一维 [N*12]
    out_data = twelve_lead.T.flatten()

    out_dir.mkdir(parents=True, exist_ok=True)
    trans_dat = out_dir / f"{record_name}_trans.DAT"
    # 转为 int16 写出（注意可能存在截断；按需可以先归一化/缩放）
    out_data.astype(np.int16).tofile(trans_dat)

    print(f'✅ {trans_dat} 已生成（int16 格式）')
    print(f'   Main导联样: {main_idx}')
    print(f'   Second导联样: {second_idx}')
    print(f'   每导联样本数: {n_samples} × 12 导联')
    return trans_dat, int(main_idx), int(second_idx)

# ------------------------------------------------
# 示例运行
# ------------------------------------------------
if __name__ == "__main__":
    process_record_to_trans_dat('./100', Path('.'))
