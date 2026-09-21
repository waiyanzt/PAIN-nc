import torch

from pain_nc.telemetry import (
    CUDA_MEMORY_KEYS,
    PeakRSSMonitor,
    cuda_memory_stats,
    model_memory_bytes,
    process_peak_rss_bytes,
    validate_resource_metrics,
)


def test_cpu_telemetry_is_complete_and_nonnegative():
    model = torch.nn.Linear(3, 2)
    memory = model_memory_bytes(model)
    gpu = cuda_memory_stats(torch.device("cpu"))
    assert memory["static_model_bytes"] == (
        memory["parameter_bytes"] + memory["buffer_bytes"]
    )
    assert set(gpu) == set(CUDA_MEMORY_KEYS)
    assert all(value == 0 for value in gpu.values())
    assert process_peak_rss_bytes() > 0
    monitor = PeakRSSMonitor().start()
    assert monitor.stop() > 0


def test_resource_contract_fails_closed():
    try:
        validate_resource_metrics({})
    except ValueError as error:
        assert "Mandatory resource telemetry is incomplete" in str(error)
    else:
        raise AssertionError("Incomplete telemetry must be rejected")


def test_complete_resource_contract_is_accepted():
    gpu = {key: 0 for key in CUDA_MEMORY_KEYS}
    resources = {
        "parameter_bytes": 1,
        "buffer_bytes": 0,
        "static_model_bytes": 1,
        "checkpoint_bytes": 2,
        "process_peak_rss_bytes": 3,
        "training_gpu": gpu,
        "inference_gpu": gpu,
        "artifacts": {"shared_bytes": 4, "variant_bytes": 5},
        "environment": {"device": "cpu"},
    }
    validate_resource_metrics(resources)
