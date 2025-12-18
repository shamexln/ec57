r"""
bxb_runner.py — 使用 WFDB 官方比较器 bxb.exe 进行逐拍对比（Beat-by-beat comparison）

功能：
- 批量（--all 或 --records）为每条记录确保生成算法 CSV 与 WFDB 注释 <record>.bio；
- 自动调用 bxb.exe 与参考注释 atr 逐拍对比；
- 解析 bxb.exe 的标准输出，抽取 TP/FP/FN（匹配/误报/漏报）等关键统计；
- 将每记录与总体的统计写入 metrics_bxb\summary.csv。

注意：
- 被测注释扩展固定为 bio（与 wfdb_runner.py 的 --to-ann 输出一致）。
- 参考扩展固定为 atr；容差由 --tolerance-ms 指定（默认 100 ms）。
- 需要已安装 wfdb（仅用于读取 fs，生成 .bio 时使用）。
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import wfdb


def list_records(db_dir: Path, records_cli: List[str] | None, use_all: bool) -> List[str]:
    if records_cli:
        return [str(r) for r in records_cli]
    
    recs = []
    if use_all:
        recs = sorted([p.stem for p in db_dir.glob('*.hea')])
    
    # 如果没找到且没显式报错，返回空列表
    return recs


def rd_fs(db_dir: Path, record: str) -> float:
    sig, hdr = wfdb.rdsamp(str((db_dir / record).as_posix()))
    return float(hdr['fs'])


def csv_to_bio(record: str, csv_path: Path, out_dir: Path, fs: float) -> Path:
    """把算法 CSV 转为 WFDB 注释 <record>.bio（兼容两种列格式）。
    列格式A（无表头）：index,time,beat,hr（首行标题，skiprows=1）。
    列格式B（有表头）：IndexPos,time,beat type,heart rate。
    使用 IndexPos 作为样本索引
    未识别 beat 默认映射为 'N'；
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bio_path = out_dir / f'{record}.bio'

    # 尝试有表头
    def _normalize(cols: List[str]) -> List[str]:
        return [str(c).strip().lower().replace(' ', '') for c in cols]

    df = None
    try:
        df0 = pd.read_csv(csv_path)
        cols = _normalize(list(df0.columns))
        df0.columns = cols
        df = df0
    except Exception:
        df = None

    samples: np.ndarray
    beats: List[str]
    if df is None or df.empty or not set(df.columns).intersection({'indexpos', 'time', 'beat', 'beattype'}):
        # 无表头格式
        df = pd.read_csv(csv_path, header=None, names=['index', 'time', 'beat', 'hr'], skiprows=1)
        if df.empty:
            # 写一个空注释文件（以免 bxb 报错）；但通常这意味着算法无输出
            wfdb.wrann(record, 'bio', sample=np.asarray([], dtype=np.int64), symbol=[])
            return bio_path
        idx = pd.to_numeric(df['indexpos'], errors='coerce').to_numpy()
        samples = np.rint(idx).astype(np.int64)
        beats = df['beat'].astype(str).tolist()
    else:
        if 'indexpos' in df.columns:
            idx = pd.to_numeric(df['indexpos'], errors='coerce').to_numpy()
            samples = np.rint(idx).astype(np.int64)
        else:
            raise ValueError(f'{csv_path} 缺少 IndexPos 或 time 列')
        bc = 'beat' if 'beat' in df.columns else ('beattype' if 'beattype' in df.columns else None)
        if bc is None:
            raise ValueError(f'{csv_path} 缺少 beat / beat type 列')
        beats = df[bc].astype(str).tolist()

    # 将 L 原样或映射？这里不改变 label，仅作为事件；bxb 只关心时间对齐
    symbols = beats

    # 排序并写出（明确写入 bio_path 指定目录）
    order = np.argsort(samples)
    samples = samples[order]
    symbols = [symbols[i] for i in order]
    # wfdb.wrann 会以当前工作目录为基准将 <record>.<ext> 写入磁盘。
    # 为确保写到 bio_path 指定目录，切换临时工作目录再写入。
    import os
    old_cwd = os.getcwd()
    try:
        os.chdir(str(out_dir))
        wfdb.wrann(record, 'bio', sample=samples, symbol=symbols)
    finally:
        os.chdir(old_cwd)
    return bio_path


def run_bxb(bxb_exe: Path, workdir: Path, record: str, wfdbpath: str, test_ext: str = 'bio') -> Tuple[str, int]:
    """调用 bxb.exe，返回标准输出文本与退出码。
    会在子进程环境中设置 WFDBPATH，以便 bxb 能找到 .hea/.dat/.atr 与 测试注释（扩展名由 test_ext 指定）。
    例如：test_ext='nbio' 或 'vbio'。
    """
    import os
    cmd = [str(bxb_exe), '-r', record, '-a', 'atr', str(test_ext), '-l' , 'sta.txt', 'ansta.txt']
    env = os.environ.copy()
    # Ensure workdir is absolute
    abs_workdir = str(workdir.resolve())
    # WFDBPATH should prioritize workdir so bxb writes output there
    env['WFDBPATH'] = f"{abs_workdir};{wfdbpath}"
    proc = subprocess.run(cmd, cwd=abs_workdir, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
    stdout = proc.stdout + (proc.stderr or '')
    return stdout, proc.returncode


def _read_bio_as_df(workdir: Path, record: str) -> pd.DataFrame:
    """读取 workdir 下 <record>.bio 为 DataFrame(sample, symbol)。若不存在返回空 DF。"""
    bio_path = workdir / f'{record}.bio'
    if not bio_path.exists():
        return pd.DataFrame({'sample': pd.Series(dtype='int64'), 'symbol': pd.Series(dtype='object')})
    ann = wfdb.rdann(str((workdir / record).as_posix()), 'bio')
    df = pd.DataFrame({'sample': np.asarray(ann.sample, dtype=np.int64), 'symbol': list(ann.symbol)})
    # 排序，去除 NaN/空符号
    df = df.dropna(subset=['sample', 'symbol'])
    df = df.sort_values('sample').reset_index(drop=True)
    return df


def _write_bio_from_df(workdir: Path, record: str, df: pd.DataFrame, out_ext: str) -> Path:
    """将 DataFrame 写为 WFDB 注释 <record>.<out_ext> 到 workdir。"""
    samples = np.asarray(pd.to_numeric(df['sample'], errors='coerce').dropna().round().astype(np.int64)) if not df.empty else np.asarray([], dtype=np.int64)
    symbols = df['symbol'].astype(str).tolist() if not df.empty else []
    import os
    old_cwd = os.getcwd()
    try:
        os.chdir(str(workdir))
        wfdb.wrann(record, out_ext, sample=samples, symbol=symbols)
    finally:
        os.chdir(old_cwd)
    return workdir / f'{record}.{out_ext}'


def parse_bxb_stdout(text: str) -> Dict[str, int | float]:
    """从 bxb 输出中抽取关键统计。不同版本格式略有区别，这里使用关键字提取：
    返回键：TP（匹配）、FP（误报）、FN（漏报）、TOTAL_REF、TOTAL_TEST。
    若无法识别，返回空字典。"""
    # 简单启发式解析
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    stats: Dict[str, int | float] = {}
    for ln in lines:
        low = ln.lower()
        # 常见关键字：matched / false positives / false negatives / total annotated / total detected
        if 'matched' in low and any(ch.isdigit() for ch in low):
            # e.g., "Matched annotations: 2274"
            digits = ''.join(ch if ch.isdigit() else ' ' for ch in ln).split()
            if digits:
                stats['TP'] = int(digits[-1])
        elif 'false positive' in low:
            digits = ''.join(ch if ch.isdigit() else ' ' for ch in ln).split()
            if digits:
                stats['FP'] = int(digits[-1])
        elif 'false negative' in low:
            digits = ''.join(ch if ch.isdigit() else ' ' for ch in ln).split()
            if digits:
                stats['FN'] = int(digits[-1])
        elif ('total' in low and 'reference' in low) or ('total' in low and 'annotated' in low):
            digits = ''.join(ch if ch.isdigit() else ' ' for ch in ln).split()
            if digits:
                stats['TOTAL_REF'] = int(digits[-1])
        elif ('total' in low and 'test' in low) or ('total' in low and 'detected' in low):
            digits = ''.join(ch if ch.isdigit() else ' ' for ch in ln).split()
            if digits:
                stats['TOTAL_TEST'] = int(digits[-1])
    return stats


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='使用 bxb.exe 对算法 .bio 与参考 .atr 做逐拍对比（批量）')
    p.add_argument('--db', required=True, help='MIT/PhysioNet 数据目录，如 C\\project\\mit-database')
    p.add_argument('--exe', default='.\\ECGAlg.exe', help='ECGAlg.exe 路径（用于生成 CSV/.bio）')
    p.add_argument('--bxb', required=True, help='bxb.exe 路径，例如 C\\project\\wfdb-10.7.0\\build\\bin\\bxb.exe')
    p.add_argument('--workdir', default='.', help='工作目录（生成 CSV/.bio 与执行 bxb 的位置）')
    p.add_argument('--outdir', default='.\\metrics_bxb', help='输出目录（summary.csv）')
    p.add_argument('--records', nargs='*', help='指定记录列表，如 100 101 103')
    p.add_argument('--all', action='store_true', help='对目录下所有 .hea 记录执行')
    p.add_argument('--tolerance-ms', type=int, default=100, help='bxb 容差（毫秒），默认 100')
    p.add_argument('--force', action='store_true', help='即使已有 CSV/.bio 也强制重跑算法与重写 .bio')
    # QRS 合并开关：默认开启，可用 --no-merge-qrs 关闭
    p.add_argument('--merge-qrs', dest='merge_qrs', action='store_true', help="将 QRS 子类 {'N','R','L','e','j','A','a','J','S'} 合并为 'N'（用于 nbio 生成）")
    p.add_argument('--no-merge-qrs', dest='merge_qrs', action='store_false', help='关闭 QRS 子类合并，仅保留原始 N 作为 Q 列')
    p.set_defaults(merge_qrs=True)
    return p.parse_args(argv)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    db_dir = Path(args.db)
    bxb_exe = Path(args.bxb)
    workdir = Path(args.workdir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    ensure_dir(workdir)

    records = list_records(db_dir, args.records, args.all)

    # 如果没有找到记录，尝试自动寻找（满足“如果没有处理数据，先代码自己调用”的要求）
    if not records:
        print(f"ℹ 在 {db_dir} 未找到记录。尝试寻找其它数据目录...")
        # 常见位置探测
        candidates = [
            Path("C:/project/mit-database"),
            db_dir.parent,
            Path(".")
        ]
        for cand in candidates:
            if cand.exists() and list(cand.glob("*.hea")):
                print(f"🔍 自动发现数据目录: {cand}")
                db_dir = cand
                records = list_records(db_dir, None, True)
                break
    
    if not records:
        print(f"⚠ 无法找到任何 .hea 记录。请通过 --db 指定正确的数据目录。")
        return 0

    # 1.5) 将最终确定的 MIT 数据目录下的所有文件（不含子目录）复制到 workdir 和 outdir
    # 以便 bxb 在工作目录直接找到 <record>.hea/.dat/.atr，以及在 outdir 存放结果
    for target in [workdir, outdir]:
        try:
            print(f"📂 正在复制数据文件到 {target} ...")
            for p in db_dir.iterdir():
                if p.is_file():
                    dst = target / p.name
                    if not dst.exists():
                        # 使用 read_bytes/write_bytes 简单实现复制
                        dst.write_bytes(p.read_bytes())
        except Exception as e:
            print(f"⚠ 复制数据文件到 {target} 时出错：{e}")

    rows: List[Dict] = []

    # 在开始循环前，清理旧的统计文件，以确保 bxb 重新开始累加（参考 process_atr.py）
    for f in ["sta.txt", "ansta.txt"]:
        f_path = workdir / f
        if f_path.exists():
            f_path.unlink()

    # 逐条记录处理
    for rec in records:
        try:
            fs = rd_fs(db_dir, rec)
            csv_path = workdir / f'{rec}.csv'
            bio_path = workdir / f'{rec}.bio'

            # 若缺 CSV 或强制，先用现有运行器产生 CSV
            if args.force or (not csv_path.exists()):
                # 复用 wfdb_runner.py 以保持一致
                runner = Path(__file__).parent / 'wfdb_runner.py'
                cmd = [
                    'python', str(runner), '--db', str(db_dir), '--record', rec,
                    '--exe', str(Path(args.exe)), '--workdir', str(workdir), '--outdir', str(workdir), '--leadnum', '12'
                ]
                subprocess.run(cmd, check=True)

            # 生成 .bio（若缺或强制）。若 db_dir 下已有同名 .bio，也会复制到 workdir。
            if not bio_path.exists():
                src_bio_in_db = db_dir / f'{rec}.bio'
                if src_bio_in_db.exists():
                    bio_path.write_bytes(src_bio_in_db.read_bytes())
                elif args.force or (not bio_path.exists()):
                    csv_to_bio(rec, csv_path, workdir, fs)
            elif args.force:
                # 强制重写
                csv_to_bio(rec, csv_path, workdir, fs)

            # 在调用 bxb 前显式检查 .bio 是否存在；若不存在则报错并给出提示
            bio_exists = bio_path.exists()
            if not bio_exists:
                rows.append({'record': rec, 'exit_code': -2, 'error': f'缺少 {bio_path}，请确认已生成 CSV 或提供现成 .bio'})
                continue

            # 调用 bxb.exe，并打印关键信息（子进程 cwd 与 WFDBPATH 会被设置）
            # Use absolute path for db_dir in WFDBPATH
            abs_db_dir = str(db_dir.resolve())
            wfdbpath = f"{abs_db_dir}"
            print(f"[bxb] cwd={workdir}")
            # 1) 整体统计（不区分类别，测试扩展使用原 bio）
            stdout_all, code_all = run_bxb(bxb_exe, workdir, rec, wfdbpath=wfdbpath, test_ext='bio')
            stats_all = parse_bxb_stdout(stdout_all)

            # 填充到 rows 中
            row = {'record': rec, 'exit_code': code_all}
            row.update(stats_all)
            rows.append(row)
        except Exception as e:
            rows.append({'record': rec, 'error': str(e), 'exit_code': -1})

    # 写出汇总
    if not rows:
        print("⚠ 没有处理任何记录，跳过汇总。")
        return 0

    df = pd.DataFrame(rows)
    df = df.sort_values('record')
    # 先写出原始明细
    (outdir / 'summary.csv').write_text(df.to_csv(index=False), encoding='utf-8')
    print(f'📄 bxb 汇总写入：{outdir / "summary.csv"}')

    # 处理累加的 sta.txt
    sta_txt = workdir / "sta.txt"
    if sta_txt.exists():
        try:
            print(f"正在读取累加统计文件: {sta_txt}")
            # 读取 sta.txt
            df_sta = pd.read_csv(sta_txt, sep=r'\s+')
            
            # 将列名映射为 Excel 列名 (A, B, C...)
            def get_excel_col_name(n):
                res = ""
                while n > 0:
                    n, rem = divmod(n - 1, 26)
                    res = chr(65 + rem) + res
                return res
            
            # 重命名列名为 A, B, C...
            excel_cols = [get_excel_col_name(i + 1) for i in range(len(df_sta.columns))]
            df_sta.columns = excel_cols

            # 对性能指标列求平均，加在最后一行（排除 - 和 0）
            # 根据用户要求，使用 Excel 列名：M, N, O, P, Q 等
            # M: Q Se, N: Q +P, O: V Se, P: V +P, Q: V FPR
            cols_to_avg = ['M', 'N', 'O', 'P', 'Q']
            # 确保这些列存在
            existing_cols = [c for c in cols_to_avg if c in df_sta.columns]
            if existing_cols:
                # 复制一份用于计算，避免修改原始数据展示
                df_calc = df_sta[existing_cols].copy()
                for c in existing_cols:
                    # 转换为数值，非数值转换为 NaN
                    df_calc[c] = pd.to_numeric(df_calc[c], errors='coerce')
                    # 将 0 替换为 NaN，以便 mean() 排除它们
                    df_calc[c] = df_calc[c].replace(0, np.nan)
                
                avg_values = df_calc.mean()
                # 创建平均值行，非目标列设为空或特定标识
                avg_row = {col: '' for col in df_sta.columns}
                if 'A' in df_sta.columns: # A 列通常是 Record
                    avg_row['A'] = 'AVERAGE'
                
                for c in existing_cols:
                    avg_row[c] = avg_values[c]
                
                df_sta = pd.concat([df_sta, pd.DataFrame([avg_row])], ignore_index=True)
                print(f"📊 已在 sta_report.csv 追加 {', '.join(existing_cols)} 的平均值行")

            # 最后将特定列名重命名为含义名
            rename_map = {
                'M': 'Q Se',
                'N': 'Q +P',
                'O': 'V Se',
                'P': 'V +P'
            }
            df_sta.rename(columns=rename_map, inplace=True)
            print(f"🏷️ 已重命名列: {rename_map}")

            output_report = outdir / "sta_report.csv"
            df_sta.to_csv(output_report, index=False, encoding='utf-8')
            print(f"✅ 累加统计报告已导出至: {output_report}")
        except Exception as e:
            print(f"⚠ 解析 sta.txt 出错: {e}")
    else:
        print(f"⚠ 未找到累加文件 {sta_txt}，请检查 bxb.exe 是否生成了该文件。")

    # 在 summary.csv 中追加 TOTAL 行（合计）
    try:
        def _num_series(name: str) -> pd.Series:
            return pd.to_numeric(df.get(name, pd.Series(dtype=float)), errors='coerce').fillna(0)

        TP_sum = int(_num_series('TP').sum())
        FP_sum = int(_num_series('FP').sum())
        FN_sum = int(_num_series('FN').sum())
        Se_sum = (TP_sum / (TP_sum + FN_sum)) if (TP_sum + FN_sum) > 0 else 0.0
        PPV_sum = (TP_sum / (TP_sum + FP_sum)) if (TP_sum + FP_sum) > 0 else 0.0

        TP_N_sum = int(_num_series('TP_N').sum())
        FP_N_sum = int(_num_series('FP_N').sum())
        FN_N_sum = int(_num_series('FN_N').sum())
        Se_N_sum = (TP_N_sum / (TP_N_sum + FN_N_sum)) if (TP_N_sum + FN_N_sum) > 0 else 0.0
        PPV_N_sum = (TP_N_sum / (TP_N_sum + FP_N_sum)) if (TP_N_sum + FP_N_sum) > 0 else 0.0

        TP_V_sum = int(_num_series('TP_V').sum())
        FP_V_sum = int(_num_series('FP_V').sum())
        FN_V_sum = int(_num_series('FN_V').sum())
        Se_V_sum = (TP_V_sum / (TP_V_sum + FN_V_sum)) if (TP_V_sum + FN_V_sum) > 0 else 0.0
        PPV_V_sum = (TP_V_sum / (TP_V_sum + FP_V_sum)) if (TP_V_sum + FP_V_sum) > 0 else 0.0

        total_row = {
            'record': 'TOTAL',
            'TP': TP_sum, 'FP': FP_sum, 'FN': FN_sum, 'Se': Se_sum, 'PPV': PPV_sum,
            'TP_N': TP_N_sum, 'FP_N': FP_N_sum, 'FN_N': FN_N_sum, 'Se_N': Se_N_sum, 'PPV_N': PPV_N_sum,
            'TP_V': TP_V_sum, 'FP_V': FP_V_sum, 'FN_V': FN_V_sum, 'Se_V': Se_V_sum, 'PPV_V': PPV_V_sum,
        }
        df2 = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
        (outdir / 'summary.csv').write_text(df2.to_csv(index=False), encoding='utf-8')
        print('➕ 已在 summary.csv 追加 TOTAL 合计行')
    except Exception as e:
        print(f'⚠ 追加 TOTAL 行失败：{e}')

    # 生成“最终报告”：按 bxb -l N 与 -l V 分别统计的结果汇总，
    # 输出到 outdir/final_report.csv，表头与 eval_atr_metrics.py 一致：
    # "", Q Se, Q +P, V Se, V +P
    try:
        # 汇总 N 类（对应 Q 列）
        TP_N_sum = int(pd.to_numeric(df.get('TP_N', pd.Series(dtype=float)), errors='coerce').fillna(0).sum())
        FP_N_sum = int(pd.to_numeric(df.get('FP_N', pd.Series(dtype=float)), errors='coerce').fillna(0).sum())
        FN_N_sum = int(pd.to_numeric(df.get('FN_N', pd.Series(dtype=float)), errors='coerce').fillna(0).sum())

        q_se = (TP_N_sum / (TP_N_sum + FN_N_sum)) if (TP_N_sum + FN_N_sum) > 0 else 0.0
        q_ppv = (TP_N_sum / (TP_N_sum + FP_N_sum)) if (TP_N_sum + FP_N_sum) > 0 else 0.0

        # 汇总 V 类
        TP_V_sum = int(pd.to_numeric(df.get('TP_V', pd.Series(dtype=float)), errors='coerce').fillna(0).sum())
        FP_V_sum = int(pd.to_numeric(df.get('FP_V', pd.Series(dtype=float)), errors='coerce').fillna(0).sum())
        FN_V_sum = int(pd.to_numeric(df.get('FN_V', pd.Series(dtype=float)), errors='coerce').fillna(0).sum())

        v_se = (TP_V_sum / (TP_V_sum + FN_V_sum)) if (TP_V_sum + FN_V_sum) > 0 else 0.0
        v_ppv = (TP_V_sum / (TP_V_sum + FP_V_sum)) if (TP_V_sum + FP_V_sum) > 0 else 0.0

        def _fmt(x: float) -> str:
            return f"{x * 100:.2f}"

        report_df = pd.DataFrame([
            ['MIT-BIH', _fmt(q_se), _fmt(q_ppv), _fmt(v_se), _fmt(v_ppv)]
        ], columns=['', 'Q Se', 'Q +P', 'V Se', 'V +P'])
        final_path = outdir / 'final_report.csv'
        report_df.to_csv(final_path, index=False, encoding='utf-8')
        print(f'📄 最终报告：{final_path}')

        # 生成可追溯详情 final_report_details.csv（百分数字符串；记录容差/合并开关）
        def _fmt_pct(x: float) -> str:
            return f"{x * 100:.2f}"

        details_rows = [
            ['Run', 'Scope', 'MIT-BIH'],
            ['Run', 'ComparatorTolerance', 'default'],
            ['Run', 'MergeQRS', str(args.merge_qrs)],
            ['Q', 'TP', str(TP_N_sum)],
            ['Q', 'FP', str(FP_N_sum)],
            ['Q', 'FN', str(FN_N_sum)],
            ['Q', 'Se', _fmt_pct(q_se)],
            ['Q', 'PPV', _fmt_pct(q_ppv)],
            ['V', 'TP', str(TP_V_sum)],
            ['V', 'FP', str(FP_V_sum)],
            ['V', 'FN', str(FN_V_sum)],
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
    raise SystemExit(main())
