"""
eval_atr_metrics.py

用途：
- 批量运行 ECGAlg.exe（可选跳过），读取算法输出 CSV，与 MIT-BIH 的参考注释 .atr 比较。
- 在给定容差（毫秒）内进行一对一拍位匹配，统计四类（N/V/F/Q）的混淆矩阵与指标。
- 生成每记录与汇总的 CSV；可选生成简单 HTML 报告（混淆矩阵热力图）。

依赖：wfdb, numpy, pandas, matplotlib（仅当 --export-html 时需要）

示例：
python eval_atr_metrics.py --db "C:\\project\\mit-database" --exe .\\ECGAlg.exe --workdir . --outdir .\\metrics --all --tolerance-ms 100 --export-html
python eval_atr_metrics.py --db "C:\\project\\mit-database" --records 100 101 103 --skip-run --outdir .\\metrics --tolerance-ms 75
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import wfdb


# 默认四类；如启用 --keep-L-class 则在 main() 中覆盖为五类
FOUR_CLASSES = ['N', 'V', 'F', 'Q']
# 运行时控制：是否保留 L 类；若不保留，L 映射到何类（N 或 Q）
KEEP_L_CLASS: bool = False
MAP_L_TO: str = 'N'

# 是否将算法端 QRS 子类合并为 'N'（避免低估 QRS 检测能力）
MERGE_QRS: bool = True
# QRS 子类集合（合并为 'N'）
QRS_SET = {'N', 'R', 'L', 'e', 'j', 'A', 'a', 'J', 'S'}


def list_records(db_dir: Path, records_cli: List[str] | None, use_all: bool) -> List[str]:
    if records_cli:
        return [str(r) for r in records_cli]
    if use_all:
        recs = []
        for p in db_dir.glob('*.hea'):
            recs.append(p.stem)
        recs.sort()
        return recs
    raise ValueError('必须使用 --records 或 --all 指定评估的记录')


def rd_fs(db_dir: Path, record: str) -> float:
    sig, hdr = wfdb.rdsamp(str((db_dir / record).as_posix()))
    return float(hdr['fs'])


def read_ref_ann(db_dir: Path, record: str) -> Tuple[np.ndarray, List[str]]:
    ann = wfdb.rdann(str((db_dir / record).as_posix()), 'atr')
    return np.asarray(ann.sample, dtype=np.int64), list(ann.symbol)


def map_ref_symbol_to_four(sym: str) -> str:
    s = str(sym)
    # 依据 AAMI 简化映射；若保留 L 类则单独输出 'L'
    n_set = {'N', 'R', 'e', 'j', 'A', 'a', 'J', 'S'}  # S族并入N
    v_set = {'V', 'E'}
    f_set = {'F'}
    if KEEP_L_CLASS and s == 'L':
        return 'L'
    if (not KEEP_L_CLASS) and s == 'L':
        return MAP_L_TO
    if s in n_set:
        return 'N'
    if s in v_set:
        return 'V'
    if s in f_set:
        return 'F'
    return 'Q'


def map_alg_symbol_to_four(sym: str) -> str:
    s = str(sym)
    # 合并策略：当 MERGE_QRS 为真时，QRS_SET ∪ {'L'} 一律映射为 'N'
    if MERGE_QRS and (s in QRS_SET):
        return 'N'
    # 若未启用合并，遵循原有映射；当保留 L 类时，L→L；否则 L→MAP_L_TO（默认N）
    if s == 'L':
        return 'L' if KEEP_L_CLASS else MAP_L_TO
    mapping = {'N': 'N', 'V': 'V', 'F': 'F', 'Q': 'Q'}
    return mapping.get(s, 'N')  # 未识别默认 N（可按需改为 'Q'）


def read_alg_csv(csv_path: Path, fs: float) -> Tuple[np.ndarray, List[str]]:
    """读取算法 CSV，兼容两种表头风格：
    1) 无表头：index,time,beat,hr（首行标题，skiprows=1）
    2) 有表头（可能含空格）：IndexPos,time,beat type,heart rate
       - 若存在 IndexPos，则直接作为样本索引（units: samples）；
       - 否则使用 time（秒）× fs 四舍五入。
    """
    # 尝试读取为有表头
    def _normalize_cols(cols: List[str]) -> List[str]:
        return [str(c).strip().lower().replace(' ', '') for c in cols]

    df = None
    try:
        df0 = pd.read_csv(csv_path)
        if not df0.empty:
            cols = _normalize_cols(list(df0.columns))
            df0.columns = cols
            df = df0
    except Exception:
        df = None

    if df is None or df.empty or (
        not set(df.columns).intersection({'IndexPos', 'time', 'beat type', 'heart rate'})
    ):
        # 回退到无表头格式
        df = pd.read_csv(csv_path, header=None, names=['index', 'time', 'beat', 'hr'], skiprows=1)
        if df.empty:
            return np.empty((0,), dtype=np.int64), []
        # 标准无表头：time 为秒
        times = pd.to_numeric(df['time'], errors='coerce').to_numpy()
        if np.isnan(times).any():
            raise ValueError(f'CSV {csv_path} time 列包含 NaN')
        samples = np.rint(times * fs).astype(np.int64)
        beats_col = 'beat'
        beats = df[beats_col].astype(str).tolist()
        beats = [map_alg_symbol_to_four(b) for b in beats]
        return samples, beats

    # 规范化表头分支
    # 优先使用 IndexPos，如果没有则使用 time × fs
    if 'indexpos' in df.columns:
        idx = pd.to_numeric(df['indexpos'], errors='coerce').to_numpy()
        if np.isnan(idx).any():
            raise ValueError(f'CSV {csv_path} IndexPos 列包含 NaN')
        samples = np.rint(idx).astype(np.int64)
    elif 'time' in df.columns:
        times = pd.to_numeric(df['time'], errors='coerce').to_numpy()
        if np.isnan(times).any():
            raise ValueError(f'CSV {csv_path} time 列包含 NaN')
        samples = np.rint(times * fs).astype(np.int64)
    else:
        raise ValueError(f'CSV {csv_path} 缺少 IndexPos 或 time 列')

    # 确定 beat 列
    beat_col = 'beat' if 'beat' in df.columns else ('beattype' if 'beattype' in df.columns else None)
    if beat_col is None:
        raise ValueError(f'CSV {csv_path} 缺少 beat / beat type 列')
    beats_raw = df[beat_col].astype(str).tolist()
    beats = [map_alg_symbol_to_four(b) for b in beats_raw]
    return samples, beats


def greedy_match(pred_samples: np.ndarray, true_samples: np.ndarray, tol_samples: int) -> Tuple[Dict[int, int], List[int], List[int]]:
    """贪心最近邻一对一匹配。
    返回：match_map（pred_idx -> true_idx），unaligned_pred_idx 列表，unaligned_true_idx 列表。
    预期 pred_samples/true_samples 已升序。
    """
    i, j = 0, 0
    match: Dict[int, int] = {}
    used_true = set()
    while i < len(pred_samples) and j < len(true_samples):
        dt = pred_samples[i] - true_samples[j]
        if abs(dt) <= tol_samples:
            # 局部在窗内，尝试在附近找最近的 true
            best_j = j
            best_abs = abs(dt)
            jj = j - 1
            while jj >= 0 and abs(pred_samples[i] - true_samples[jj]) <= tol_samples:
                if abs(pred_samples[i] - true_samples[jj]) < best_abs and jj not in used_true:
                    best_j = jj
                    best_abs = abs(pred_samples[i] - true_samples[jj])
                jj -= 1
            jj = j + 1
            while jj < len(true_samples) and abs(pred_samples[i] - true_samples[jj]) <= tol_samples:
                if abs(pred_samples[i] - true_samples[jj]) < best_abs and jj not in used_true:
                    best_j = jj
                    best_abs = abs(pred_samples[i] - true_samples[jj])
                jj += 1
            if best_j not in used_true:
                match[i] = best_j
                used_true.add(best_j)
            i += 1
            # j 前移到未使用的最近处
            while j < len(true_samples) and j in used_true:
                j += 1
        elif dt < 0:
            # 预测在参考之前，且超出窗；该预测不可能与当前 j 匹配
            i += 1
        else:
            # 参考在预测之前，且超出窗；推进参考索引
            j += 1

    unaligned_pred = [pi for pi in range(len(pred_samples)) if pi not in match]
    unaligned_true = [ti for ti in range(len(true_samples)) if ti not in used_true]
    return match, unaligned_pred, unaligned_true


def confusion_and_metrics(y_true: List[str], y_pred: List[str],
                          unmatched_pred_classes: List[str], unmatched_true_classes: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    # 混淆矩阵（行：真，列：预测）
    cm = pd.DataFrame(0, index=FOUR_CLASSES, columns=FOUR_CLASSES, dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in FOUR_CLASSES and p in FOUR_CLASSES:
            cm.loc[t, p] += 1

    # 将未配对的预测/参考计入 FP/FN
    extra_fp = {c: 0 for c in FOUR_CLASSES}
    for c in unmatched_pred_classes:
        if c in extra_fp:
            extra_fp[c] += 1
    extra_fn = {c: 0 for c in FOUR_CLASSES}
    for c in unmatched_true_classes:
        if c in extra_fn:
            extra_fn[c] += 1

    # 逐类 TP/FP/FN/TN（包含未配对带来的 FP/FN）
    per_cls = []
    # 总体样本数按“参考拍总数 + 额外被算法误报的拍数（unmatched_pred）”估算
    total = cm.values.sum() + sum(extra_fn.values()) + sum(extra_fp.values())
    for c in FOUR_CLASSES:
        TP = cm.loc[c, c]
        FP = (cm[c].sum() - TP) + extra_fp[c]
        FN = (cm.loc[c].sum() - TP) + extra_fn[c]
        TN = total - TP - FP - FN
        sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        f1 = (2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0
        acc = (TP + TN) / total if total > 0 else 0.0
        per_cls.append({'class': c, 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
                        'Sensitivity': sens, 'Specificity': spec, 'PPV': prec, 'F1': f1, 'Accuracy': acc})
    per_cls_df = pd.DataFrame(per_cls)

    # 微/宏平均（包含未配对的 FP/FN）
    TP_sum = sum(cm.loc[c, c] for c in FOUR_CLASSES)
    FP_sum = sum((cm[c].sum() - cm.loc[c, c]) + extra_fp[c] for c in FOUR_CLASSES)
    FN_sum = sum((cm.loc[c].sum() - cm.loc[c, c]) + extra_fn[c] for c in FOUR_CLASSES)
    TN_sum = total - TP_sum - FP_sum - FN_sum
    micro_recall = TP_sum / (TP_sum + FN_sum) if (TP_sum + FN_sum) > 0 else 0.0
    micro_prec = TP_sum / (TP_sum + FP_sum) if (TP_sum + FP_sum) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_recall / (micro_prec + micro_recall)) if (micro_prec + micro_recall) > 0 else 0.0
    micro_acc = TP_sum / total if total > 0 else 0.0
    micro_spec = TN_sum / (TN_sum + FP_sum) if (TN_sum + FP_sum) > 0 else 0.0

    macro = per_cls_df[['Sensitivity', 'Specificity', 'PPV', 'F1', 'Accuracy']].mean().to_dict()
    summary = {
        'Micro_Recall': micro_recall,
        'Micro_Specificity': micro_spec,
        'Micro_Precision': micro_prec,
        'Micro_F1': micro_f1,
        'Micro_Accuracy': micro_acc,
        'Macro_Recall': macro['Sensitivity'],
        'Macro_Specificity': macro['Specificity'],
        'Macro_Precision': macro['PPV'],
        'Macro_F1': macro['F1'],
        'Macro_Accuracy': macro['Accuracy'],
        'Total': total,
    }
    return cm, per_cls_df, summary


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def maybe_run_algorithm(db_dir: Path, record: str, exe: Path, workdir: Path, outdir: Path, force: bool) -> Path:
    """调用 wfdb_runner.py 生成 CSV，返回 CSV 路径。"""
    csv_path = workdir / f'{record}.csv'
    if csv_path.exists() and not force:
        return csv_path
    # 调用同目录的 wfdb_runner.py
    import subprocess, sys
    runner = Path(__file__).parent / 'wfdb_runner.py'
    # 为与 C:\\project\\ECGAlg 的参考实现一致：
    # - 将 trans.dat 写入工作目录下 example_data 子目录
    # - settings.ini 使用相对路径 .\\example_data\\<record>_trans.dat
    example_outdir = workdir / 'example_data'
    example_outdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(runner), '--db', str(db_dir), '--record', str(record), '--exe', str(exe), '--workdir', str(workdir), '--outdir', str(example_outdir), '--leadnum', '12']
    subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
    return csv_path


def export_html_report(outdir: Path, overall_cm: pd.DataFrame, per_record_rows: List[Dict]):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(overall_cm.values, cmap='Blues')
        ax.set_xticks(range(len(FOUR_CLASSES)))
        ax.set_xticklabels(FOUR_CLASSES)
        ax.set_yticks(range(len(FOUR_CLASSES)))
        ax.set_yticklabels(FOUR_CLASSES)
        for i in range(len(FOUR_CLASSES)):
            for j in range(len(FOUR_CLASSES)):
                ax.text(j, i, str(overall_cm.values[i, j]), ha='center', va='center', color='black')
        ax.set_title('Overall Confusion Matrix (N/V/F/Q)')
        fig.colorbar(im, ax=ax, shrink=0.8)
        buf = BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png', dpi=160)
        plt.close(fig)
        img_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

        html = [
            '<html><head><meta charset="utf-8"><title>ECG Evaluation Report</title></head><body>',
            '<h2>Overall Confusion Matrix (N/V/F/Q)</h2>',
            f'<img src="data:image/png;base64,{img_b64}"/>',
            '<h2>Per-record Metrics (Micro Accuracy & F1)</h2>',
            '<table border="1" cellspacing="0" cellpadding="4">',
            '<tr><th>Record</th><th>Total</th><th>Micro Accuracy</th><th>Micro F1</th></tr>'
        ]
        for row in per_record_rows:
            html.append(f"<tr><td>{row['record']}</td><td>{row['Total']}</td><td>{row['Micro_Accuracy']:.4f}</td><td>{row['Micro_F1']:.4f}</td></tr>")
        html.append('</table></body></html>')
        out_html = outdir / 'report.html'
        out_html.write_text('\n'.join(html), encoding='utf-8')
    except Exception as e:
        (outdir / 'report_error.txt').write_text(str(e), encoding='utf-8')


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='批量评估 ECGAlg.exe 与 MIT-BIH .atr 的一致性（四类 N/V/F/Q）')
    p.add_argument('--db', required=True, help='本地 WFDB 数据目录，例如 C:\\project\\mit-database')
    p.add_argument('--exe', default='.\\ECGAlg.exe', help='ECGAlg.exe 路径；配合 --skip-run 可不需要')
    p.add_argument('--workdir', default='.', help='算法运行与 CSV 输出目录（record.csv）')
    p.add_argument('--outdir', default='.\\metrics', help='评估输出目录（CSV/HTML）')
    p.add_argument('--records', nargs='*', help='指定记录列表，例如 100 101 103')
    p.add_argument('--all', action='store_true', help='评估目录下所有 .hea 记录')
    p.add_argument('--tolerance-ms', type=int, default=100, help='匹配容差（毫秒），常用 100 或 75')
    p.add_argument('--force', action='store_true', help='强制重跑算法（即使 CSV 已存在）')
    p.add_argument('--skip-run', action='store_true', help='跳过运行算法，直接评估已存在的 CSV')
    p.add_argument('--export-html', action='store_true', help='导出简单 HTML 报告（混淆矩阵热力图）')
    p.add_argument('--keep-L-class', action='store_true', help='评估时保留 L 为独立类别（形成 N/L/V/F/Q 五类）')
    p.add_argument('--map-L-to', choices=['N', 'Q'], default='N', help='当不保留 L 类时，L 映射到哪个类别（默认 N）')
    # 新增：是否将算法端 QRS 子类合并为 N（避免低估 QRS 能力）
    p.add_argument('--merge-qrs', dest='merge_qrs', action='store_true', help="将算法端 QRS 子类 {'N','R','L','e','j','A','a','J','S'} 合并为 'N'")
    p.add_argument('--no-merge-qrs', dest='merge_qrs', action='store_false', help='关闭算法端 QRS 子类合并（按原始 N/V/F/Q 使用）')
    p.set_defaults(merge_qrs=True)
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    db_dir = Path(args.db)
    exe = Path(args.exe)
    workdir = Path(args.workdir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    per_record_dir = outdir / 'per_record'
    ensure_dir(per_record_dir)

    # 运行期类别与映射策略
    global FOUR_CLASSES, KEEP_L_CLASS, MAP_L_TO, MERGE_QRS
    KEEP_L_CLASS = bool(args.keep_L_class)
    MAP_L_TO = str(args.map_L_to)
    MERGE_QRS = bool(args.merge_qrs)
    if KEEP_L_CLASS:
        FOUR_CLASSES = ['N', 'L', 'V', 'F', 'Q']

    records = list_records(db_dir, args.records, args.all)
    per_record_rows: List[Dict] = []
    overall_cm = pd.DataFrame(0, index=FOUR_CLASSES, columns=FOUR_CLASSES, dtype=int)

    for rec in records:
        try:
            fs = rd_fs(db_dir, rec)
            csv_path = workdir / f'{rec}.csv'
            if not args.skip_run:
                csv_path = maybe_run_algorithm(db_dir, rec, exe, workdir, outdir, args.force)
            if not csv_path.exists():
                print(f'⚠ 跳过 {rec}：未找到 {csv_path}')
                continue

            # 读取数据
            ref_samples, ref_syms = read_ref_ann(db_dir, rec)
            ref_cls = [map_ref_symbol_to_four(s) for s in ref_syms]
            pred_samples, pred_syms = read_alg_csv(csv_path, fs)
            # 将算法端符号按策略映射为评估类别
            pred_cls = [map_alg_symbol_to_four(s) for s in pred_syms]

            # 升序排序
            ref_order = np.argsort(ref_samples)
            ref_samples = ref_samples[ref_order]
            ref_cls = [ref_cls[i] for i in ref_order]
            pred_order = np.argsort(pred_samples)
            pred_samples = pred_samples[pred_order]
            pred_cls = [pred_cls[i] for i in pred_order]

            tol_samp = int(round(args.tolerance_ms * fs / 1000.0))
            match, unpred, untrue = greedy_match(pred_samples, ref_samples, tol_samp)

            # 构造已配对样本的标签
            y_true, y_pred = [], []
            for pi, tj in match.items():
                y_true.append(ref_cls[tj])
                y_pred.append(pred_cls[pi])

            # 未配对的预测拍（FP）和参考拍（FN）的类别
            unmatched_pred_classes = [pred_cls[i] for i in unpred]
            unmatched_true_classes = [ref_cls[j] for j in untrue]

            cm, per_cls_df, summary = confusion_and_metrics(
                y_true, y_pred, unmatched_pred_classes, unmatched_true_classes
            )
            overall_cm = overall_cm.add(cm, fill_value=0).astype(int)

            # 保存每记录结果
            cm_path = per_record_dir / f'{rec}_confusion.csv'
            cls_path = per_record_dir / f'{rec}_per_class.csv'
            cm.to_csv(cm_path, encoding='utf-8')
            per_cls_df.to_csv(cls_path, index=False, encoding='utf-8')

            row = {'record': rec}
            row.update(summary)
            per_record_rows.append(row)

            print(f'✅ {rec} 完成：Micro_Acc={summary["Micro_Accuracy"]:.4f} Micro_F1={summary["Micro_F1"]:.4f} TotalPairs={summary["Total"]}')

        except Exception as e:
            print(f'❌ {rec} 失败：{e}')

    # 汇总
    summary_df = pd.DataFrame(per_record_rows)
    sum_path = outdir / 'summary_metrics.csv'
    overall_cm_path = outdir / 'overall_confusion.csv'
    summary_df.to_csv(sum_path, index=False, encoding='utf-8')
    overall_cm.to_csv(overall_cm_path, encoding='utf-8')
    print(f'📄 汇总写入：{sum_path} ; 混淆矩阵：{overall_cm_path}')

    # 在 summary_metrics.csv 末追加 TOTAL 行（包含 Q/V 的 TP/FP/FN 合计与派生指标），方便快速查阅
    try:
        def _tp_fp_fn_of(cls: str) -> tuple[int, int, int]:
            if cls not in overall_cm.index or cls not in overall_cm.columns:
                return 0, 0, 0
            TP = int(overall_cm.loc[cls, cls])
            FN = int(overall_cm.loc[cls].sum() - TP)
            FP = int(overall_cm[cls].sum() - TP)
            return TP, FP, FN

        TP_Q, FP_Q, FN_Q = _tp_fp_fn_of('Q')
        TP_V, FP_V, FN_V = _tp_fp_fn_of('V')
        Se_Q = (TP_Q / (TP_Q + FN_Q)) if (TP_Q + FN_Q) > 0 else 0.0
        PPV_Q = (TP_Q / (TP_Q + FP_Q)) if (TP_Q + FP_Q) > 0 else 0.0
        Se_V = (TP_V / (TP_V + FN_V)) if (TP_V + FN_V) > 0 else 0.0
        PPV_V = (TP_V / (TP_V + FP_V)) if (TP_V + FP_V) > 0 else 0.0

        total_row = {
            'record': 'TOTAL',
            'TP_Q': TP_Q, 'FP_Q': FP_Q, 'FN_Q': FN_Q, 'Se_Q': Se_Q, 'PPV_Q': PPV_Q,
            'TP_V': TP_V, 'FP_V': FP_V, 'FN_V': FN_V, 'Se_V': Se_V, 'PPV_V': PPV_V,
        }
        sum_df2 = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)
        sum_path.write_text(sum_df2.to_csv(index=False), encoding='utf-8')
        print('➕ 已在 summary_metrics.csv 追加 TOTAL 合计行（Q/V 指标）')
    except Exception as e:
        print(f'⚠ 追加 TOTAL 行失败：{e}')

    if args.export_html:
        export_html_report(outdir, overall_cm, per_record_rows)
        print(f'🖼 报告：{(outdir / "report.html").resolve()}')

    # 生成“最终报告”文件，按照用户所需格式：
    #            Q Se			Q +P			V Se			V +P
    # MIT-BIH    99.79			99.83			90.69			92.69
    try:
        # 从 overall_cm 计算每类的 Se 与 +P（PPV）
        # overall_cm 的行是真值，列是预测
        def _se_ppv_of(cls: str) -> tuple[float, float]:
            if cls not in overall_cm.index or cls not in overall_cm.columns:
                return 0.0, 0.0
            TP = int(overall_cm.loc[cls, cls])
            FN = int(overall_cm.loc[cls].sum() - TP)
            FP = int(overall_cm[cls].sum() - TP)
            se = (TP / (TP + FN)) if (TP + FN) > 0 else 0.0
            ppv = (TP / (TP + FP)) if (TP + FP) > 0 else 0.0
            return se, ppv

        q_se, q_ppv = _se_ppv_of('Q')
        v_se, v_ppv = _se_ppv_of('V')

        # 按百分数两位小数输出
        def _fmt(x: float) -> str:
            return f"{x * 100:.2f}"

        report_df = pd.DataFrame(
            [[
                'MIT-BIH',
                _fmt(q_se), _fmt(q_ppv),
                _fmt(v_se), _fmt(v_ppv),
            ]],
            columns=['', 'Q Se', 'Q +P', 'V Se', 'V +P']
        )
        final_path = outdir / 'final_report.csv'
        report_df.to_csv(final_path, index=False, encoding='utf-8')
        print(f'📄 最终报告：{final_path}')

        # 生成可追溯详情 final_report_details.csv（百分数字符串；记录容差/合并开关）
        def _fmt_pct(x: float) -> str:
            return f"{x * 100:.2f}"

        # 计算 TP/FP/FN（与上方 Se/PPV 一致）
        def _tp_fp_fn_of2(cls: str) -> tuple[int, int, int]:
            if cls not in overall_cm.index or cls not in overall_cm.columns:
                return 0, 0, 0
            TP = int(overall_cm.loc[cls, cls])
            FN = int(overall_cm.loc[cls].sum() - TP)
            FP = int(overall_cm[cls].sum() - TP)
            return TP, FP, FN

        TP_Q, FP_Q, FN_Q = _tp_fp_fn_of2('Q')
        TP_V, FP_V, FN_V = _tp_fp_fn_of2('V')

        details_rows = [
            ['Run', 'Scope', 'MIT-BIH'],
            ['Run', 'Tolerance(ms)', str(int(args.tolerance_ms))],
            ['Run', 'MergeQRS', str(MERGE_QRS)],
            ['Q', 'TP', str(TP_Q)],
            ['Q', 'FP', str(FP_Q)],
            ['Q', 'FN', str(FN_Q)],
            ['Q', 'Se', _fmt_pct(q_se)],
            ['Q', 'PPV', _fmt_pct(q_ppv)],
            ['V', 'TP', str(TP_V)],
            ['V', 'FP', str(FP_V)],
            ['V', 'FN', str(FN_V)],
            ['V', 'Se', _fmt_pct(v_se)],
            ['V', 'PPV', _fmt_pct(v_ppv)],
        ]
        details_df = pd.DataFrame(details_rows, columns=['Section', 'Item', 'Value'])
        details_path = outdir / 'final_report_details.csv'
        details_df.to_csv(details_path, index=False, encoding='utf-8')
        print(f'📄 详情报告：{details_path}')
    except Exception as e:
        (outdir / 'final_report_error.txt').write_text(str(e), encoding='utf-8')

    return 0


if __name__ == '__main__':
    import sys
    raise SystemExit(main(sys.argv[1:]))
