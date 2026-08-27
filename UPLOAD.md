# 上传到 4open.science 的步骤

论文中引用的匿名仓库 URL：

**https://anonymous.4open.science/r/HRAG-DocAudit**

## 当前状态

本地导出包已就绪（约 5.7 MB）：

```
submission/eaai/HRAG-DocAudit-export/
```

已执行 `git init` 并完成本地 commit；远程仓库 **尚未在 4open.science 上创建**，因此自动 `git push` 失败（404）。

## 手动上传（约 2 分钟）

1. 打开 [https://anonymous.4open.science/](https://anonymous.4open.science/)
2. 点击 **New repository**，名称填 **`HRAG-DocAudit`**
3. 选择 **Upload files** 或按页面提示关联 Git remote
4. 上传整个 `HRAG-DocAudit-export/` 目录内容（或解压 `HRAG-DocAudit-export.zip` 后上传）

## 或使用 Git（创建仓库后）

```bash
cd submission/eaai/HRAG-DocAudit-export
git remote add origin https://anonymous.4open.science/r/HRAG-DocAudit.git
git push -u origin master
```

## 包内可复现命令

```bash
cd replication
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_datasets.sh   # 下载 C3PA / ContractNLI
python3 scripts/generate_cnas_holdout.py
bash run_all.sh
python3 run_tier2_baselines.py
python3 scripts/generate_calibration_curve.py
python3 scripts/sync_manuscript_from_results.py
```

## 结果文件

| 文件 | 内容 |
|------|------|
| `replication/results/tier2_baselines.json` | Self-RAG / ReAct C3PA 代理 + Tier-2 映射 |
| `replication/results/calibration.json` | ECE 0.26 → 0.07（Platt hold-out） |
| `replication/config/deployment_anchor.json` | 合作实验室 Tier-2 锚点指标 |
