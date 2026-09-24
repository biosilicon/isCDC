"""GPU monitoring failures retain observations without authorizing stale admissions."""

import threading
from types import SimpleNamespace

import pytest

from iscdc import spatialglue_monitor as monitor


def fake_nvml(monkeypatch, *, unavailable=False):
    calls = {"initializations": 0, "process_reads": 0}

    def initialize():
        calls["initializations"] += 1
        return 0

    def handle(uuid, pointer):
        calls["uuid"] = uuid
        pointer._obj.value = 123
        return 0

    def memory(handle, pointer):
        assert handle.value == 123
        pointer._obj.total = 80 * 2**30
        pointer._obj.free = 70 * 2**30
        pointer._obj.used = 10 * 2**30
        return 0

    def utilization(handle, pointer):
        pointer._obj.gpu = 43
        return 0

    def processes(handle, count, rows):
        assert handle.value == 123
        calls["process_reads"] += 1
        capacity = count._obj.value
        count._obj.value = 70
        if capacity < 70:
            return 7
        for index in range(70):
            rows[index].pid = index + 100
            rows[index].usedGpuMemory = (index + 1) * 2**20
        if unavailable:
            rows[0].usedGpuMemory = monitor.ct.c_ulonglong(-1).value
        return 0

    library = SimpleNamespace(
        nvmlInit_v2=initialize,
        nvmlShutdown=lambda: 0,
        nvmlDeviceGetHandleByUUID=handle,
        nvmlDeviceGetMemoryInfo=memory,
        nvmlDeviceGetUtilizationRates=utilization,
        nvmlDeviceGetComputeRunningProcesses_v2=processes,
    )
    monkeypatch.setattr(monitor.ct, "CDLL", lambda name: library)
    return calls


def test_nvml_reads_bytes_from_selected_device_and_growing_process_list(monkeypatch):
    calls = fake_nvml(monkeypatch)
    reader = monitor.NVMLReader("example")
    result = reader.read()
    assert result["free_bytes"] == 70 * 2**30
    assert result["utilization_percent"] == 43
    assert result["process_bytes"][100] == 2**20
    assert len(result["process_bytes"]) == 70
    assert calls == {"initializations": 1, "process_reads": 2, "uuid": b"GPU-example"}
    reader.read()
    assert calls["initializations"] == 1
    reader.close()


def test_unavailable_process_memory_is_not_reported_as_zero(monkeypatch):
    fake_nvml(monkeypatch, unavailable=True)
    reader = monitor.NVMLReader("GPU-example")
    with pytest.raises(RuntimeError, match="unavailable"):
        reader.read()
    reader.close()


def test_monitor_retains_last_sample_but_rejects_failure_and_staleness(monkeypatch):
    # Drive the actual sampler loop without wall-clock sleeps or a GPU.
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    now = [100.0]
    monkeypatch.setattr(monitor.time, "monotonic", lambda: now[0])
    current = monitor.GPUMonitor("GPU-example")
    assert not current.snapshot()["fresh"]
    assert current.snapshot()["free_bytes"] is None
    values = iter(
        [
            {"free_bytes": 70 * 2**30, "utilization_percent": 12, "process_bytes": {1: 2**30}},
            RuntimeError("temporary read failure"),
            {"free_bytes": 68 * 2**30, "utilization_percent": 22, "process_bytes": {1: 3 * 2**30}},
        ]
    )

    def read():
        value = next(values)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(
        monitor, "NVMLReader", lambda uuid: SimpleNamespace(read=read, close=lambda: None)
    )
    snapshots = []

    def next_iteration(seconds):
        snapshots.append(current.snapshot())
        if len(snapshots) == 1:
            now[0] += 6
            stale = current.snapshot()
            assert not stale["fresh"] and stale["sample_age_seconds"] == 6
        if len(snapshots) == 3:
            current.stopped.set()

    monkeypatch.setattr(current.stopped, "wait", next_iteration)
    current._run()
    assert snapshots[0]["fresh"]
    assert not snapshots[1]["fresh"]
    assert "temporary read failure" in snapshots[1]["error"]
    assert snapshots[1]["process_bytes"] == {1: 2**30}
    assert snapshots[2]["fresh"] and snapshots[2]["error"] is None
    assert snapshots[2]["process_bytes"] == {1: 3 * 2**30}
