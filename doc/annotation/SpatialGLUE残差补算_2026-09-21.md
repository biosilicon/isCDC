# SpatialGLUE 甲基化残差组合补算（2026-09-21）

启动补算时按用户决定，仅在 **node3** 后台补算，**发布暂缓**。node1 上已发布的 201 个
Database、207 个组合保持原版本；启动补算阶段未执行发布或网站重启。

## 原因与修复

唯一失败组合为 `GSE270498_spatial_dmt_me11_replicate_50um` 的 `rna__methylation`，
包含 1,947 个观测、48,440 个 RNA 特征及 87,789 个甲基化特征。
canonical metadata 与 S032 接入记录明确该甲基化矩阵为 signed MethSCAn residual。
此前只支持 `[0,1]` 比例的校验错误拒绝了这份输入。2026-09-20 的只读扫描记录范围为
−1.6054529315673347 至 1.4399391197846938，未发现非有限值。

该已核实来源采用 `methylation_residual_v1`：保留符号和观测零值，标准化后 PCA，
报告记录来源语义和与比例切片不可数值比较的提示。其他甲基化来源继续执行比例范围
校验，所有来源仍拒绝 NaN/Inf。canonical H5MU 与 catalogue 均未改写。

本轮针对来源限定、残差输入/预处理、非有限值拒绝和受体零值保留的 3 项窄范围回归
检查通过，相关代码 Ruff 检查通过。未增加资源基准或运行完整测试套件。

## 后台任务

- 主机：`yzy-node3`（10.138.46.173），NVIDIA A800 80GB PCIe。
- 运行目录：`temp/spatialglue_repair_20260921/`。
- 启动时间：2026-09-21 07:47:15 UTC；控制器 PID `3099887`，批次 PID `3099994`。
- 冻结清单仅包含上述一个组合，使用完整 600 epochs；甲基化配方已确认是残差配方。
- 源 H5MU SHA-256：`e602948309534ed882d21bf8df2ff788eb8781069b7f97b92168d2ee6c7bec7c`。
- 结果保存在运行目录的 `sidecars/`，不会自动合入正式目录。

监控命令（在 node3 执行）：

```bash
ISCDC_CONDA_SH=/home1/shezixi/miniconda3/etc/profile.d/conda.sh \
  bash annotation/spatial_domain/gpu/run_full.sh status \
  --run-root temp/spatialglue_repair_20260921 --watch

tail -F temp/spatialglue_repair_20260921/run.log
```

`run_state.json` 为总状态，`attempts/*/progress.json` 记录当前阶段、任务和资源，
`attempts/*/job_*/run.log` 保存算法输出；最终成功以 `state=complete`、`successful=1`
及对应通过验证的 sidecar 为准。

最初的自动发布包装程序在启动独立控制器后，因网站环境不含 `psutil` 而退出；计算
控制器独立运行，不受包装程序退出影响。用户随后明确暂缓发布，
`temp/spatialglue_repair_publish_20260921.sh` 已禁用，未保留自动发布等待进程。
网站实际运行在 node1（10.138.46.171），其 HTTP 健康检查正常，当前 node3 的本机端口
不代表线上网站状态。将来发布须另行获得授权并在正确部署主机执行。

## 完成核验与上线准备（2026-09-21）

按用户“检查新计算的一片是否已经完成，并准备上线”的要求，在 node1 完成只读发布
审计。计算于 **07:51:17 UTC** 正常结束，`state=complete`、`total=successful=1`、
`remaining=0`；批次进度记录 `failed=0`、`pending=0`、`active_count=0`。

- 结果 generation：`20260921T075111-deab0aae2994`。
- 样本 `E11_replicate`：1,947 个观测全部完成分析，10 个空间域，`Not analyzed=0`。
- 训练日志包含 `600/600`，结果报告为 `passed`；无样本级运行警告。
- 甲基化使用 `methylation_residual_v1`，保留 signed MethSCAn residual 的符号；报告保留
  与甲基化比例不可数值比较的说明。

10:06:16 UTC，以下完整审计通过，选中 1 个组合，无暂缓或跳过项：

```bash
bash annotation/spatial_domain/publish_completed.sh \
  --method spatialglue --run-root temp/spatialglue_repair_20260921 --check-only
```

源 H5MU SHA-256、当前 catalogue 绑定、冻结文件、代码适配器、环境锁、完整参数及
sidecar 覆盖、标签、点位和 QC 均通过校验。证据位于
`temp/spatial_domain_publication_20260921T100612Z-6a9fb07d/audit.json`。
另在 node1 核实 `deploy_test.sh status` 返回健康；沙箱内本机 IPC 被拒绝时的
“not running”输出不能用作服务停机依据。

独立候选目录位于 `temp/spatialglue_release_prepare_20260921/candidate/sidecars/`。
暂存校验通过，逐文件 SHA-256 对比确认原有 **3,612 个文件全部保留且内容不变**，
仅新增该组合的 10 个文件，无修改或删除。候选版本含 202 个 SpatialGLUE Database、
208 个组合，原有 217 份 RNA 结果保留。差异凭据为同一准备目录下的
`publication_delta.json`、`baseline_hashes.json` 和 `candidate_hashes.json`。

准备阶段的正式版本（本次发布的回滚基线）为
`data/.spatial_domain_visualizations-releases/spatial_domain_publication_20260920T165330Z-9a0ed440/sidecars/`。

准备阶段未切换正式指针或重启网站。发布命令如下（随后按用户追加授权在 **node1**
实际执行）；它会重新审计、合入结果、切换目录、重启并核验，失败自动回滚：

```bash
bash annotation/spatial_domain/publish_completed.sh \
  --method spatialglue --run-root temp/spatialglue_repair_20260921
```

本批次已全部成功，无需 `--completed-only`。旧自动发布包装脚本继续保持禁用。

## 正式上线（10:21 UTC）

用户随后明确要求“停止检查，直接上线”。已停止额外的离线应用检查，直接执行上述
正式发布入口；保留发布脚本内置的来源、暂存、健康状态及线上结果验证。额外离线检查
未完成，不将其计为通过；最终上线成功依据为正式发布脚本的验证与成功凭据。

发布于 **10:21:32 UTC** 成功完成，进程退出码为 0。10:20:07 UTC 切换正式目录，
网站重启成功，新组合的详情页 generation、样本、点位 MIME、长度、SHA-256 以及
网站健康状态和前端 bundle 均通过脚本内置验证。

本次新增 `GSE270498_spatial_dmt_me11_replicate_50um` 的 `rna__methylation` 一个组合，
generation 为 `20260921T075111-deab0aae2994`。SpatialGLUE 累计覆盖 202 个 Database、
208 个组合；发布流程保留已有 RNA 结果和其他组合，没有改写源 H5MU 或 catalogue。

- 发布凭据：`temp/spatial_domain_publication_20260921T101849Z-5ad81412/published.json`。
- 审计、事件和服务日志：同目录的 `audit.json`、`events.jsonl`、`service.log`。
- 完整终端日志：`temp/spatialglue_repair_publication_20260921.log`。
- 正式 release：
  `data/.spatial_domain_visualizations-releases/spatial_domain_publication_20260921T101849Z-5ad81412/sidecars/`。
- 旧 release 保留为回滚基线；未清理原计算结果或独立准备目录。
