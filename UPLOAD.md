# 上传到 4open.science（Anonymous GitHub）

论文引用 URL：**https://anonymous.4open.science/r/HRAG-DocAudit-D8AE**

## 当前进度

| 步骤 | 状态 |
|------|------|
| 本地复现包 | ✅ `HRAG-DocAudit-export/`（5.7 MB） |
| GitHub 源仓库（公开） | ✅ https://github.com/13923870749/HRAG-DocAudit |
| 4open 匿名镜像 | ✅ **https://anonymous.4open.science/r/HRAG-DocAudit-D8AE**（Status: Ready） |

> 说明：首次尝试创建的 `HRAG-DocAudit` slug 因未授权而损坏；4open 自动分配后缀 `-D8AE`。论文链接已同步更新。

## 一键完成匿名镜像（约 1 分钟）

1. 打开 **[anonymous.4open.science/anonymize](https://anonymous.4open.science/anonymize)**
2. 点击 **Sign in with GitHub** → 在 GitHub 页点击 **Authorize**（需已启用 2FA）
3. 填写表单：
   - **GitHub URL**: `https://github.com/13923870749/HRAG-DocAudit`
   - **Branch**: `master`
   - **Anonymized repository ID**: `HRAG-DocAudit`
   - **Terms to redact**（每行一条）:
     ```
     刘辉
     Hui Liu
     雷琼钰
     Qiongyu Lei
     冯锐
     Rui Feng
     37352366@qq.com
     175543208@qq.com
     2140747@qq.com
     13923870749
     Shenzhen Information Security Management Center
     深圳
     scutliu37352366
     ```
4. 点击 **Anonymize Repository**
5. 验证：打开 https://anonymous.4open.science/r/HRAG-DocAudit-D8AE 应能看到 `README.md` 和 `replication/`

## 本地复现（无需 4open）

```bash
cd replication
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_datasets.sh
python3 scripts/generate_cnas_holdout.py
bash run_all.sh
python3 run_tier2_baselines.py
python3 scripts/generate_calibration_curve.py
```

## 备用：ZIP 直传

若 OAuth 受阻，可将 `../HRAG-DocAudit-export.zip`（536 KB）作为 Editorial Manager 补充材料上传；正文链接仍建议使用 4open 匿名 URL。

## 源仓库更新

```bash
cd HRAG-DocAudit-export
git push github master
# 然后在 4open 仪表盘点击 Update / Auto update
```
