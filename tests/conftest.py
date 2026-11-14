"""Cấu hình pytest chung.

1. Hàm stub `raise NotImplementedError(...)` → test được báo SKIPPED ("Chưa cài: ..."), không FAIL.
   Cài xong hàm là test tự chạy thật.
2. Test đánh dấu @pytest.mark.gpu tự bỏ qua khi không có torch + CUDA.
"""
import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if call.when == "call" and call.excinfo is not None and call.excinfo.errisinstance(NotImplementedError):
        rep.outcome = "skipped"
        rep.longrepr = (str(item.fspath), item.location[1] or 0, f"Chưa cài: {call.excinfo.value}")


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if _has_cuda():
        return
    skip = pytest.mark.skip(reason="cần GPU (torch + CUDA)")
    for it in items:
        if "gpu" in it.keywords:
            it.add_marker(skip)
