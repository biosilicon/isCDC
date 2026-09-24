# SpatialGLUE 后台 GPU 监控修复（2026-09-20）

原运行目录为 `temp/spatialglue_full/`，任务范围为 202 个数据集、208 个组合。
首轮在 11:37:22 UTC 的整卡查询、重试轮在 11:43:14 UTC 的进程查询分别发生
`subprocess.TimeoutExpired`；11:45:09 UTC 总控以 `incomplete` 退出，完成数 0/208。
两轮曾启动 80 个工作进程，但均未进入正式计算阶段，不能视为 80 个任务正在训练。

## 查询核对

用户反馈终端 `nvidia-smi` 一直正常。在原 Conda＋GPU 环境中复用完整命令、GPU UUID、
`timeout=10`，整卡／进程查询各重复三次，全部返回 0，stderr 为空：

| 查询 | 三次耗时（秒） |
| --- | --- |
| `--query-gpu=memory.free,utilization.gpu -i GPU-60dd3110-9f3d-ae43-e7c4-46c5246a8296` | 0.058、0.061、0.017 |
| `--query-compute-apps=pid,used_gpu_memory` | 0.040、0.038、0.038 |

两条查询均带 `--format=csv,noheader,nounits`。实际二进制为 `/usr/bin/nvidia-smi`，
`LD_LIBRARY_PATH`／`LD_PRELOAD` 未设置，二进制与驱动 NVML 库均位于本机 XFS。
未发现查询参数错误，现有证据不能认定驱动或 `nvidia-smi` 损坏，也不能确认当时单次超时
的底层原因。此前看到 Python 进程等待 NFS，不足以证明这两次 GPU 查询超时由 NFS 引起。

恢复后，在 80 个工作进程已启动的实际负载下，读取调度器环境并分别使用原始命令名和
`/usr/bin/nvidia-smi` 绝对路径执行相同查询。四次也全部成功：整卡为 0.045／0.064 秒，
进程为 0.025／0.041 秒。没有复现路径查找错误或查询超时。
原始输出保存在新运行目录的 `nvidia_smi_comparison.json`。

确认的实现缺陷是：调度器每秒同步启动两次查询，硬编码 10 秒超时；异常直接冒泡到整轮
清理逻辑，使一次监控失败终止全部工作进程。一次短暂读取失败因此丢失整轮在途计算。

## 修复和验证

- 通过进程内 NVML 句柄读取指定 GPU 的空闲显存、利用率和进程显存，取消反复启动命令。
  ABI 对照 [NVIDIA NVML 头文件](https://github.com/NVIDIA/go-nvml/blob/main/pkg/nvml/nvml.h)，
  使用匹配的 `nvmlProcessInfo_v2_t` 与版本化 `_v2` 函数。
- 单个后台线程每秒采样；调度循环不等待采样调用。读取失败或采样超过 5 秒未更新时，
  保留已运行任务，暂停新任务／新阶段准入，恢复后继续；采样年龄从读取开始时计时。
- 保留最后一次观测但标记不可用于准入，心跳和状态显示后端、采样年龄及错误。
  未知进程显存不按零处理，进程列表增长时按 NVML 返回数量扩容重读。
- Ruff 检查通过；监控的三个回归用例和两个既有调度用例共 5 项通过。
  A800 单次只读 NVML 检查成功：空闲显存 73.035 GiB，返回 4 个 GPU 进程。
  本次没有追加吞吐量或资源规模实验，也没有修改训练参数、任务上限及预留预算。

## 恢复入口

旧冻结目录及日志完整保留。新运行使用原 `temp/spatialglue_full/catalog.db` 快照，冻结
修复后的执行代码，输出到 `temp/spatialglue_full_nvml_20260920/`；旧 sidecar 仅在校验通过后
复用。

新总控 PID 为 `3572245`，调度器 PID 为 `3572740`。截至 12:01 UTC，80 个工作进程已启动，
任务开始进入 `pairing` 阶段，128 项排队；NVML 样本正常更新且无监控错误。

监控新运行时显式指定目录：

```bash
annotation/spatial_domain/gpu/run_full.sh status --watch --run-root temp/spatialglue_full_nvml_20260920
tail -f temp/spatialglue_full_nvml_20260920/run.log
```

停止／续跑同样使用此 `--run-root`。计算结果仍不自动发布或重启网站。
