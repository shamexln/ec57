EC57 — WFDB Runner & Evaluator 使用说明 (中文/English)

—— 简介 ——
本项目提供两个脚本：
- wfdb_runner.py：从本地 MIT/PhysioNet WFDB 数据读取记录，重排为标准 12 导联，生成 <record>_trans.dat 与 settings.ini，并调用 ECGAlg.exe 运行算法；可选将算法 CSV 转为 WFDB 注释 .bio。
- eval_atr_metrics.py：批量将算法输出与 MIT‑BIH 参考注释 .atr 对齐评估，计算 N/V/F/Q 四类混淆矩阵与各项指标，并导出 CSV/HTML 报告。

工作目录默认：C:\project\EC57（可在任意目录使用）。
MIT 数据目录可配置，示例默认为：C:\project\mit-database。

—— 快速开始（≤ 5 行）——
1) 可选：python -m venv .venv && .\.venv\Scripts\Activate.ps1
2) pip install wfdb numpy pandas matplotlib
3) 单记录运行：python wfdb_runner.py --db "C:\\project\\mit-database" --record 100 --exe .\ECGAlg.exe --workdir . --outdir . --to-ann
4) 批量评估：python eval_atr_metrics.py --db "C:\\project\\mit-database" --exe .\ECGAlg.exe --workdir .\run_out --outdir .\run_out --all 
5) 结果查看：metrics\summary_metrics.csv 与 metrics\report.html

—— 环境要求 ——
- Windows（脚本已按 Windows 路径与 PowerShell 示例给出）
- Python 3.8+（建议 3.9/3.10）
- 依赖：wfdb、numpy、pandas、matplotlib（HTML 报告需要）
- 可执行文件：ECGAlg.exe（放在当前目录或用 --exe 指定）
- 数据：MIT‑BIH Arrhythmia Database（本地路径示例 C:\project\mit-database，包含 .hea/.dat/.atr）

—— wfdb_runner.py 用法 ——
用途：
- 读取指定记录（如 100），将导联重排为标准 12 导联顺序：I, II, III, aVR, aVL, aVF, V1–V6（缺失导联零填充）。
- 自动选择 MainLead/SecondLead（优先 II/MLII 与 V1，若缺失则退化）
- 生成 <record>_trans.dat（int16, little-endian；按“样本×12导联”展平）与 settings.ini
- 调用 ECGAlg.exe 生成 <record>.csv；可选将 CSV 转为 WFDB 注释 <record>.bio

命令示例：
1) 运行并生成注释：
   python wfdb_runner.py --db "C:\\project\\mit-database" --record 100 --exe .\ECGAlg.exe --workdir . --outdir . --to-ann

2) 仅生成数据与 settings.ini，不运行算法：
   python wfdb_runner.py --db "C:\\project\\mit-database" --record 101 --outdir . --workdir . --skip-run

常用参数：
- --db <DIR>           MIT 数据目录（可配置；示例 C:\\project\\mit-database）
- --record <ID>        记录号（如 100、101）
- --exe <PATH>         ECGAlg.exe 路径（默认 .\ECGAlg.exe）
- --workdir <DIR>      生成 settings.ini 与算法输出 CSV 的目录（默认当前）
- --outdir <DIR>       输出 <record>_trans.dat 的目录（默认当前）
- --to-ann             将 CSV 转换为 WFDB 注释 <record>.bio
- --skip-run           不执行 ECGAlg.exe，仅准备输入文件

输出文件：
- <record>_trans.dat   12 导联打包信号（int16，小端）
- settings.ini         ECGAlg.exe 配置
- <record>.csv         ECGAlg.exe 输出（若执行）
- <record>.bio         由 CSV 转为 WFDB 注释（启用 --to-ann 时）

—— eval_atr_metrics.py 用法 ——
评测原理：将算法 CSV 的拍点统一到样本时间轴，与参考 .atr 在 ±容差窗口内进行一对一最近邻匹配；配对样本计入混淆矩阵，未配对的算法/参考分别记为 FP/FN。类别采用 AAMI 简化四类（S 并入 N）：N/V/F/Q。

典型命令：
1) 评估全部记录，默认 ±100 ms，导出 HTML：
   python eval_atr_metrics.py --db "C:\\project\\mit-database" --exe .\ECGAlg.exe --workdir . --outdir .\metrics --all --tolerance-ms 100 --export-html

2) 评估指定记录（±75 ms），并强制重跑算法：
   python eval_atr_metrics.py --db "C:\\project\\mit-database" --exe .\ECGAlg.exe --workdir . --outdir .\metrics --records 100 101 103 --tolerance-ms 75 --force

3) 直接评估已存在的 CSV（不再调用算法）：
   python eval_atr_metrics.py --db "C:\\project\\mit-database" --workdir . --outdir .\metrics --records 100 101 --skip-run

生成内容：
- metrics\per_record\<record>_confusion.csv  每记录的 4×4 混淆矩阵
- metrics\per_record\<record>_per_class.csv  每记录各类 TP/FP/FN/TN 与指标
- metrics\overall_confusion.csv              汇总混淆矩阵
- metrics\summary_metrics.csv                总体与宏/微平均指标
- metrics\report.html                        可选 HTML 报告（含热力图）

常用参数：
- --all / --records <ID...>  批量选择
- --tolerance-ms <INT>       匹配容差（默认 100，建议 75~100）
- --force                    即使存在 CSV 也重跑算法
- --skip-run                 跳过算法，仅评估现有 CSV
- --export-html              生成 HTML 报告（需 matplotlib）

—— ECGAlg.exe 参数/版本说明 ——
settings.ini 关键项（示例，与算法保持一致；脚本会自动写入合理值）：
- InputDataPath   输入数据路径（<record>_trans.dat，建议相对路径，相对于 --workdir）
- OutputDataPath  输出 CSV 路径（<record>.csv，建议相对路径）
- Fs              采样率（如 360）
- ADperMV         标度（默认 200，可根据算法要求修改 wfdb_runner.py 的 write_settings 逻辑）
- LeadNum         导联数（固定 12，用于 12 导联输入；如算法按 2 导联工作，仍由脚本写入 Main/Second 以指示主次导联）
- MainLead        主导联（1 基，优先 II/MLII；脚本自动选择）
- SecondLead      次导联（1 基，优先 V1；脚本自动选择）
- 其他：ADorCh/PacePatient/LeadType/Norch 等保持与你现有工程一致。

版本建议：
- 将 ECGAlg.exe 与其版本号/日期一并保存在仓库或同目录（例如 ECGAlg_2024-12-01.exe），或记录在 CHANGELOG 中。
- 若程序支持命令行 --version，请执行并在 README 中补充输出；若不支持，可以文件时间或内置文档标注版本。

—— 常见问题（FAQ）——
Q1: 报错找不到记录 100（No such record）。
A: 确认 --db 指向的目录下存在 100.hea/100.dat；或该目录为 MIT‑BIH 的根目录。

Q2: 运行 ECGAlg.exe 失败或无输出 CSV。
A: 确认 --exe 路径正确并有执行权限；检查 settings.ini 的 InputDataPath/OutputDataPath 是否为相对 --workdir 的可写路径。

Q3: HTML 报告未生成。
A: 安装 matplotlib：pip install matplotlib；或去掉 --export-html 参数。

Q4: 混淆矩阵/指标看起来异常。
A: 检查容差设置（--tolerance-ms 过小会导致错配）；确认算法 CSV 列顺序与脚本假设相符（index,time,beat,hr，首行标题会被跳过）。

Q5: 导联映射是否固定？
A: 是，目标顺序固定为 I, II, III, AVR, AVL, AVF, V1–V6。若原记录缺失某些导联将零填充；主次导联自动选择。

—— 备注 ——
- 若希望将未识别算法符号从默认的 N 改为 Q，可编辑 eval_atr_metrics.py 中 map_alg_symbol_to_four 的默认返回。
- 可将容差常用档位预设为 75/100 ms，通过命令行切换。

====================================================================
English (Brief)

Overview
- wfdb_runner.py: Read local MIT/PhysioNet records, remap to standard 12-lead layout, build <record>_trans.dat and settings.ini, run ECGAlg.exe, optionally convert CSV to WFDB annotations (.bio).
- eval_atr_metrics.py: Batch evaluation against MIT‑BIH .atr, one-to-one matching within a tolerance window, metrics for 4 classes (N/V/F/Q), export CSV/HTML.

Quick Start (≤ 5 lines)
1) Optional venv: python -m venv .venv && .\.venv\Scripts\Activate.ps1
2) pip install wfdb numpy pandas matplotlib
3) Single record: python wfdb_runner.py --db "C:\\project\\mit-database" --record 100 --exe .\ECGAlg.exe --workdir . --outdir . --to-ann
4) Batch eval:    python eval_atr_metrics.py --db "C:\\project\\mit-database" --exe .\ECGAlg.exe --workdir . --outdir .\metrics --all --tolerance-ms 100 --export-html
5) See results:   metrics\summary_metrics.csv, metrics\report.html

Configurable paths
- MIT data directory is configurable via --db (default example: C:\project\mit-database).
- ECGAlg.exe path via --exe (default .\ECGAlg.exe).

Outputs
- <record>_trans.dat, settings.ini, <record>.csv, <record>.bio (optional)
- Per-record and overall CSV metrics, optional HTML report

ECGAlg.exe notes
- settings.ini keys: InputDataPath, OutputDataPath, Fs, ADperMV, LeadNum=12, MainLead/SecondLead (1-based). Keep other flags consistent with your binary.
- Versioning: keep the exe with a version tag or record the date; use --version if supported.

FAQ
- Ensure records exist under --db; check permissions and relative paths against --workdir; install matplotlib for HTML.
