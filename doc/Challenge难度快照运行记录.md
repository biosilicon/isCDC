# Challenge 难度快照运行记录

本文件记录本地目录的日期化发布结果。评估方法和使用命令见
[README](../README.md#challenge-distribution-shift-难度参考)，快照契约见
[数据库存储规范 1.2](数据库存储规范_v1.2.md)。运行快照、输入数据、备份和详细审计均不纳入
版本控制；新 checkout 需为自己的目录生成快照。

## 2026-09-05：恢复 entry-ID 修正后的难度展示

历史 entry-ID 修正重写了 train/test H5MU，旧快照保存的 SHA-256 与当前 catalogue 不一致，
应用因而将难度信息显示为 `Unavailable` / `null`。本次重新运行固定 domain classifier，
对全部 29 个 Challenge 重新计算指标与全局排名，严格校验后原子替换正式快照，并重启服务。
截至本次验收，该问题已解决。

### 评估与目录基线

| 项目 | 本次记录 |
| --- | --- |
| Catalogue | v5；242 个正式文件：184 full、29 train、29 test；30 个不同 entry_id |
| 评估结果 | 29 个 Challenge，29 成功、0 失败 |
| 输入文件 | 58 个 train/test H5MU，实际 SHA-256 全部与 catalogue 一致 |
| 输入与采样 | RNA，seed 42，每侧最多 5,000 个有效 observation |
| 重复与交叉验证 | 5 次重采样 × 5 折，共 725 折 |
| Representation / classifier | 每折训练部分选择最多 2,000 个高方差共同特征；whitened 50 维 PCA；固定 L2 logistic regression |
| Mean AUROC 范围 | 0.5490564166179068–1.0 |
| 与旧结果比较 | 29 项 mean AUROC 全部相同，最大绝对差为 0 |
| 正式发布时间 | 2026-09-05 04:31:32 UTC |
| 线上验收完成时间 | 2026-09-05 04:32:09 UTC |

本次是完整重新评估，结果与旧指标相同；发布的新快照记录了当前输入文件的 SHA-256。
AUROC 相同不免除文件身份校验，也不能据此将以后过期的快照直接视为有效。

报告保留以下方法诊断，均未导致评估失败：

| Warning code | 出现次数 |
| --- | ---: |
| `low_sample_size` | 3 |
| `hierarchy_domain_aligned` | 14 |
| `small_category_pool` | 2 |
| `preprocessed_input` | 1 |

出现次数按 warning 记录统计，同一 Challenge 可以有多条诊断。解释指标时仍需考虑样本量、
层级混杂、类别规模和输入预处理；这些指标仅表示 train/test 分布可分性。

### 发布验收与制品身份

- 候选快照通过应用使用的 `load_difficulty_snapshot` 严格校验，覆盖全部 29 个 Challenge，
  核对类型、输入模态、两侧 ID/SHA-256、指标和全局百分位一致性；另核验默认参数及每项 25 折结果。
- 正式快照通过 `write_report_atomically(..., force=True)` 发布，发布后重新加载校验通过。
- 通过 `deploy_test.sh restart` 重启原有 iscdc 服务，健康检查通过。
- 69 个线上 HTTP 响应全部通过：29 个详情 API、29 个详情页面，以及健康检查、列表和分页
  请求；29 项 API 指标与新快照精确一致，页面按规定精度展示，升降序和分页结果均正确。
- 评估和发布前后 catalogue SHA-256 不变。未修改项目代码，未运行完整测试套件。

| 制品 | SHA-256 |
| --- | --- |
| `data/catalog.db` | `4c9ada1f3cf0ebe8a5758b481573ddadea6cee55f09e19a278e58ec6f2f2b5ae` |
| `data/challenge_difficulty.json` | `12cd0e40070bd7214ffff05012a81b52352de999d128403ad9a7a31d06f1d275` |

本地审计位于 `temp/difficulty_recompute_20260905T042101Z/`：

- `baseline.json`：目录哈希、旧快照失效原因及 58 个输入文件身份。
- `challenge_difficulty.before.json`：旧快照备份。
- `evaluation.log`：输入哈希检查、逐 Challenge 进度及 CLI 成功结果。
- `challenge_difficulty.candidate.json`：完成评估的候选快照。
- `publication_audit.json`、`live_verification.json`：发布与线上验收证据。
- `publish.py`、`verify_live.py`、`completion.md`：本次发布/核验脚本及本地完成记录。

此前 `temp/approved_entry_import_handoff_20260904.md` 中“旧 difficulty 快照仍不可用”的描述
属于当时状态；本次发布已解决该项，旧交接文件保留历史内容。
