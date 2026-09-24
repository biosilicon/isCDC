# SpatialGLUE 接入与隔离验证（2026-09-20）

本次在原 RNA BANKSY/GraphST 结果旁新增独立的 SpatialGLUE 结果，支持含 RNA 的两、三模态
组合，四模态数据须显式选择子集。正式 catalogue、源 H5MU 和线上 sidecar 不在本次修改范围；
实际产物、校准脚本、日志、浏览器证据保存在 `temp/spatialglue_20260920/`。
方法、默认参数、资源限制和命令见 [空间域识别](空间域识别.md#spatialglue-多模态空间域)。

## 实现边界

- 每个 sample 按 observation ID 独立配对，取已选模态均存在且非零的交集；其他观测保留为
  `Not analyzed`，分别记录缺失模态和全零原因。不插补缺失模态，不改写源标签。
- RNA 使用 HVG/PCA，蛋白使用 CLR/PCA，ATAC 和组蛋白使用 TF-IDF/LSI。模型输出经 PCA、
  邻居图和直接调用 igraph 后端的 Leiden 聚类，不依赖 R 或 mclust，不预设目标域数。
- 官方训练器会覆盖部分构造参数，适配器在训练前重设已解析的 epochs 和损失权重。
  空间图明确使用行 ID 并排除自身，支持坐标重合；特征图采用 correlation 距离。
- SpatialGLUE 使用 manifest/report v2 和独立 `methods/spatialglue` 状态。旧 RNA v1、旧 URL、
  公开 catalogue API 保持兼容。页面在同一画布中切换三类结果，保留相机和样本状态，图例独立。

## 环境

环境声明及完整锁文件位于 `annotation/spatial_domain/gpu/`，本机实际安装目录为
`temp/environments/iscdc-spatial-domain-gpu`。采用 Python 3.11、PyTorch 2.5.1+cu121、
SpatialGlue 1.1.5、SpatialGlue_3M 0.0.2、Scanpy 1.10.4、igraph 1.0.0。
安装后 `pip check` 通过。CUDA 确定性算法开启，固定 seed 42，关闭 TF32。

设备为 NVIDIA A800 80GB PCIe，驱动 550.107.02，UUID
`GPU-60dd3110-9f3d-ae43-e7c4-46c5246a8296`。环境锁、适配器、源数据、各阶段和产物摘要
均写入实际 generation；运行资源以实测报告为准。

## 验证命令

```bash
# 网站回归（iscdc 环境）
PYTHONPATH=src python -m pytest tests/test_spatial_domain_visualization.py \
  tests/test_spatialglue.py tests/test_spatial_domain_batch.py \
  tests/test_spatial_domain_parallel.py tests/test_spatial_domain_publish.py

# GPU 集成（上述 GPU 环境，必须能访问 A800）
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 \
  python -m pytest annotation/spatial_domain/gpu/tests/test_inference.py

# 真实样本校准，先串行，再依据结果决定下一步
PYTHONPATH=src python -u temp/spatialglue_20260920/profile_runs.py --phase serial
```

GPU 集成包含实际官方双模态和三模态训练器的短轮次训练及重放；真实样本使用完整默认轮数。
本次验证范围是计算完整性、重放及页面行为，不代表生物学真值准确率验证。

## 测量后确定并行数

先完成 SPOTS RNA＋蛋白和 Spatial-Mux-seq RNA＋蛋白＋H3K27me3 两个完整训练任务的串行
测量。端到端耗时分别 135.79、288.06 秒，总计 423.85 秒；包含子进程的峰值 RSS 分别
1.42、1.69 GiB，PyTorch 峰值保留显存分别 102、132 MiB。资源余量足够后，才进行双任务
对照；总耗时 287.27 秒，吞吐量为串行的 1.475 倍，两任务峰值 RSS 之和为 3.09 GiB。
据此，本次剩余五个代表数据集采用 **2 个并行任务，每任务 8 线程**，各分配 64 GiB 主存、
32 GiB 显存预算。没有推断更高并行数的收益，也没有启用全量后台任务。

资源租约由校准脚本持有，子进程继承全局资源锁和 GPU UUID 锁，CPU affinity 不重叠。
主存/显存上限为任务预算，不是预分配量。单任务和双任务均保留 8 GiB 可用显存门槛。
`nvidia-smi` 是整卡采样，包含已有进程占用；不可将其直接解释成本任务显存。
串行阶段最初的 NVML 采样未正确补上 `GPU-` UUID 前缀，随后补充独立采样；因此不报告
覆盖整个串行阶段的平均 GPU 利用率。每个任务自身的 RSS 和 PyTorch 峰值显存记录完整。
双任务及后续阶段的整卡采样已修正。

重放比较的源摘要、配对顺序、预处理结果和模型输入图摘要相同，最终 assignments 完全一致。
双模态和三模态联合表示最大绝对差分别为 1.27×10⁻⁷、1.49×10⁻⁷，通过
`atol=1e-6, rtol=1e-5`。联合表示、PCA 和聚类图的字节摘要存在差异，不能称为逐位一致；
报告保留这些差异，不以四舍五入覆盖原始产物。

对应证据为 `calibration_serial/profile.json`、`calibration_parallel/profile.json`、
`concurrency.json` 和 `replay.json`。重放与后续命令：

```bash
PYTHONPATH=src python -u temp/spatialglue_20260920/profile_runs.py --phase parallel
python temp/spatialglue_20260920/compare_replay.py
PYTHONPATH=src python -u temp/spatialglue_20260920/profile_runs.py --phase remaining --workers 2
```

## 自动化与页面验证

网站相关 5 个测试文件共 **92 项通过**；GPU 集成 **14 项通过**；Node 24 前端单元测试
**25 项通过**；Chromium 真实页面检查 **3 次通过**（两个测试场景，含四模态子集复查）；`make lint`、GPU 测试文件 lint 和
`git diff --check` 通过。未运行完整 pytest 套件。日志分别为 `website-tests.log`、
`gpu-tests.log`、`browser-tests.log` 和 `browser-four-modalities.log`。

浏览器检查覆盖旧 RNA/细胞类型切换，以及 SpatialGLUE 的同一画布复用、相机位置、样本、
独立图例、方法弹窗和默认隐藏的未分析类。检查时修复了鼠标离开画布后异步拾取回调重新
显示悬浮提示的问题。相机比较严格比对绘图区像素，裁掉会因弹窗滚动锁定产生轻微抗锯齿
差异的 CSS 圆角；图中点的像素保持一致。

桌面 1440×1000 和手机 390×844 均无页面横向溢出、无浏览器控制台错误，各只有一个画布。
图像及结构检查证据为 `desktop-spatialglue.png`、`mobile-spatialglue.png` 和
`visual-qa.json`。预览使用独立 catalogue 副本、sidecar 根目录和关闭的 analytics；不重启
正式服务。

复查隔离页面可在 `iscdc` 环境使用以下命令（仅绑定本机端口）：

```bash
PYTHONPATH=src \
ISCDC_DATABASE_PATH="$PWD/temp/spatialglue_20260920/catalog.db" \
ISCDC_SPATIAL_DOMAIN_VISUALIZATION_ROOT="$PWD/temp/spatialglue_20260920/sidecars" \
ISCDC_CELL_TYPE_VISUALIZATION_ROOT="$PWD/temp/spatialglue_20260920/cells" \
ISCDC_ANALYTICS_DATABASE_PATH="$PWD/temp/spatialglue_20260920/analytics.db" \
ISCDC_ANALYTICS_ENABLED=0 \
python -m uvicorn iscdc.app:app --host 127.0.0.1 --port 8768
```

## 七个代表数据集的结果

全部成功，合计 31,063 个观测。以下是隔离汇总目录所选 generation 的推理耗时和峰值；
SPOTS 和 RNA＋蛋白＋H3K27me3 两个校准案例保留串行 generation，其余五个使用双任务
阶段产物。耗时不含父进程启动、
最终 sidecar 提交等少量开销，因此与上述端到端校准时间不同。

| Database ID | 已选模态 | 分析观测 | 域数 | 推理秒数 | 峰值 RSS / GiB | CUDA reserved / MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gse198353_spots_mouse_spleen_rep1 | rna + protein | 2653 | 7 | 131.1 | 1.42 | 102.0 |
| GSE205055_mouse_brain_p22_20um_atac_rna | rna + atac | 9215 | 13 | 537.6 | 6.18 | 210.0 |
| GSE279771_human_gbm_space_seq_atac_rna | rna + atac | 2768 | 9 | 505.4 | 5.24 | 108.0 |
| GSE205055_mouse_brain_p21_20um_h3k27ac_rna | rna + histone | 2387 | 9 | 287.3 | 1.54 | 108.0 |
| GSE263333_mouse_embryo_e13_50um_sample3_atac_rna_h3k4me3 | rna + atac + histone | 2133 | 10 | 515.9 | 9.5 | 136.0 |
| gse263333_spatial_mux_seq_mouse_embryo_e13_20um_h3k27me3_rna_protein | rna + protein + histone | 2221 | 18 | 281.9 | 1.66 | 132.0 |
| GSE263333_mouse_brain_5m_20um_spatial_mux_h3k27ac | rna + protein + atac | 9686 | 10 | 381.3 | 5.09 | 294.0 |

七个真实样本均为完整配对、非零输入；缺失模态和零计数边界由独立 GPU fixture 测试覆盖。
GBM 案例验证 ATAC binary，其余 ATAC 案例为 counts。最后一行使用四模态原数据的显式
RNA＋蛋白＋ATAC 子集，`unused_modalities` 为 `histone`。

每个任务均核对源 H5MU 的前后 SHA-256；源摘要、generation ID、实际特征数及逐样本资源
汇总见 `results.json`，完整依据保留在各 generation 中。统一只读审计 `audit.json` 验证
**7 套 RNA + 7 套 SpatialGLUE 均为 success**，两套方法各自的状态和产物共存。

最终隔离结果位于 `temp/spatialglue_20260920/sidecars/`。本次未向正式 sidecar 根目录
发布，未修改源 H5MU 或正式 catalogue，未重启线上服务。

四模态子集页面额外验证通过：方法说明显示 `Unused modalities: histone`，切换后相机、
样本与独立图例保持正确。截图/trace 保存在 `browser-four-modalities-evidence/`。
本次临时预览在检查完成后关闭，可按上述命令重新启动。
