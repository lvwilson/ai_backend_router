"""GPU-free process-guard and unload tests using disposable OS processes."""
import asyncio
import os
import signal
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from router import create_app
from src.config import RouterConfig
from src.orchestrator import Orchestrator
from src.process_guard import ProcessGuardError
from src.service_loader import ServiceConfig, ServiceState


@pytest.fixture(autouse=True)
def no_hogs(monkeypatch):
    monkeypatch.setattr(
        "src.orchestrator.query_per_process_vram_with_names", AsyncMock(return_value=[])
    )


@pytest.fixture
def processes():
    children = []

    def spawn():
        child = subprocess.Popen(["sleep", "60"], start_new_session=True)
        children.append(child)
        return child

    yield spawn
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


@pytest.fixture
async def orch():
    instance = Orchestrator(
        [ServiceConfig(name="model", binary="unused")], total_vram_gb=48
    )
    yield instance
    await instance.process_guards.close()


async def wait_until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(poll(), 3)


def fake_running(orch, monkeypatch, stop_gate=None):
    loader = orch.services["model"]
    loader._state = ServiceState.RUNNING
    loader._process = SimpleNamespace(pid=99999999, returncode=None)
    loader._actual_vram_gb = 4
    orch.note_extra_vram("model", 8)

    async def stop():
        if stop_gate is not None:
            await stop_gate.wait()
        loader._process.returncode = 0
        loader._state = ServiceState.DEAD
        return True

    monkeypatch.setattr(loader, "stop", AsyncMock(side_effect=stop))
    monkeypatch.setattr(loader, "save_slot_cache", AsyncMock())
    return loader


async def test_natural_exit_releases_all_model_slots(orch, processes):
    child = processes()
    guard = orch.process_guards.register(child.pid)
    assert guard["blocking"] and guard["timeout"] is None
    tasks = [asyncio.create_task(orch.acquire_slot(n)) for n in ("model", "audio", "video")]
    await asyncio.sleep(0.03)
    assert not any(t.done() for t in tasks)
    assert orch.busy_backends() == []
    child.terminate()
    # Do not reap before the guard observes exit: zombies must count as finished.
    await asyncio.wait_for(asyncio.gather(*tasks), 3)
    assert orch.process_guards.snapshots() == []
    for name in ("model", "audio", "video"):
        await orch.release_slot(name)


async def test_timeout_releases_without_killing_and_retains_registration(orch, processes):
    child = processes()
    orch.process_guards.register(child.pid, timeout=0.05)
    await asyncio.wait_for(orch.acquire_slot("model"), 3)
    assert child.poll() is None
    assert orch.process_guards.snapshots()[0]["state"] == "timed_out"
    assert child.pid in orch.process_guards.pids
    await orch.release_slot("model")
    # A timed-out but registered process can still be explicitly killed.
    assert await orch.process_guards.kill(child.pid) == {"pid": child.pid, "killed": True}
    assert child.wait(timeout=2) == -signal.SIGKILL
    assert not orch.process_guards.blocking


async def test_all_guards_must_finish(orch, processes):
    first, second = processes(), processes()
    orch.process_guards.register(first.pid, 0.02)
    orch.process_guards.register(second.pid)
    pending = asyncio.create_task(orch.acquire_slot("model"))
    await wait_until(lambda: orch.process_guards.snapshots()[0]["state"] == "timed_out")
    assert not pending.done()
    await orch.process_guards.unregister(second.pid)
    await asyncio.wait_for(pending, 2)
    assert first.poll() is None and second.poll() is None
    await orch.release_slot("model")


async def test_cancel_guard_wait_does_not_leak_backend_semaphore(orch, processes):
    child = processes()
    orch.process_guards.register(child.pid)
    pending = asyncio.create_task(orch.acquire_slot("model"))
    await asyncio.sleep(0.02)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await orch.process_guards.unregister(child.pid)
    await asyncio.wait_for(orch.acquire_slot("model"), 1)
    await orch.release_slot("model")


async def test_direct_startup_also_obeys_guard(orch, processes, monkeypatch):
    child = processes()
    orch.process_guards.register(child.pid)
    ensure = AsyncMock(return_value=orch.services["model"])
    monkeypatch.setattr(orch, "_ensure_running_locked", ensure)
    pending = asyncio.create_task(orch.ensure_running("model"))
    await asyncio.sleep(0.02)
    ensure.assert_not_awaited()
    await orch.process_guards.unregister(child.pid)
    assert await asyncio.wait_for(pending, 1) is orch.services["model"]
    assert orch.busy_backends() == []


async def test_unload_drains_and_blocks_new_requests(orch, monkeypatch):
    gate = asyncio.Event()
    loader = fake_running(orch, monkeypatch, gate)
    await orch.acquire_slot("model")
    unloading = asyncio.create_task(orch.unload_all())
    await wait_until(lambda: orch._unloading == 1)
    new_request = asyncio.create_task(orch.acquire_slot("model"))
    await asyncio.sleep(0.02)
    loader.stop.assert_not_awaited()
    assert not new_request.done()
    await orch.release_slot("model")
    await wait_until(lambda: loader.stop.await_count == 1)
    assert not new_request.done()
    gate.set()
    assert await unloading == ["model"]
    await asyncio.wait_for(new_request, 1)
    loader.save_slot_cache.assert_awaited_once()
    assert orch._extra_vram == {}
    assert loader.actual_vram_gb is None
    await orch.release_slot("model")
    assert await orch.unload_all() == []


async def test_unload_allowed_during_guard(orch, processes, monkeypatch):
    child = processes()
    orch.process_guards.register(child.pid)
    fake_running(orch, monkeypatch)
    assert await asyncio.wait_for(orch.unload_all(), 1) == ["model"]
    assert orch.process_guards.blocking
    assert child.poll() is None


async def test_cancel_unload_while_draining_reopens_admission(orch):
    await orch.acquire_slot("model")
    pending = asyncio.create_task(orch.unload_all())
    await wait_until(lambda: orch._unloading == 1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert orch._unloading == 0
    await orch.release_slot("model")
    await asyncio.wait_for(orch.acquire_slot("model"), 1)
    await orch.release_slot("model")


async def test_cancel_unload_during_stop_finishes_cleanup(orch, monkeypatch):
    gate = asyncio.Event()
    loader = fake_running(orch, monkeypatch, gate)
    pending = asyncio.create_task(orch.unload_all())
    await wait_until(lambda: loader.stop.await_count == 1)
    pending.cancel()
    await asyncio.sleep(0.02)
    assert orch._unloading == 1
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert not loader.is_alive
    assert orch._unloading == 0


async def test_concurrent_unloads_are_serialized(orch, monkeypatch):
    loader = fake_running(orch, monkeypatch)
    results = await asyncio.gather(orch.unload_all(), orch.unload_all())
    assert results == [["model"], []]
    loader.stop.assert_awaited_once()


async def test_failed_stop_is_not_reported_as_unloaded(orch, monkeypatch):
    loader = fake_running(orch, monkeypatch)
    monkeypatch.setattr(loader, "stop", AsyncMock(return_value=False))
    with pytest.raises(RuntimeError, match="Failed to unload"):
        await orch.unload_all()
    assert orch._unloading == 0
    assert orch._extra_vram["model"] == 8


async def test_guard_is_excluded_from_hogs_and_group_kills(orch, processes, monkeypatch):
    child = processes()
    orch.process_guards.register(child.pid)
    other = 99999998
    monkeypatch.setattr(
        "src.orchestrator.query_per_process_vram_with_names",
        AsyncMock(return_value=[(child.pid, 20, "sleep"), (other, 10, "external")]),
    )
    assert await orch.detect_vram_hogs() == [(other, 10, "external")]
    # The unrelated hog happens to share a process group with the guard.
    with monkeypatch.context() as patch:
        patch.setattr(os, "getpgid", Mock(return_value=123))
        killpg = Mock()
        patch.setattr(os, "killpg", killpg)
        assert await orch.kill_vram_hogs() == 0
        killpg.assert_not_called()


async def test_kill_uses_pinned_handle_not_pid_or_process_group(orch, processes, monkeypatch):
    child = processes()
    orch.process_guards.register(child.pid)
    send = Mock(wraps=signal.pidfd_send_signal)
    with monkeypatch.context() as patch:
        patch.setattr(signal, "pidfd_send_signal", send)
        numeric_kill = Mock(side_effect=AssertionError("must use pidfd"))
        patch.setattr(os, "kill", numeric_kill)
        patch.setattr(os, "killpg", numeric_kill)
        await orch.process_guards.kill(child.pid)
        send.assert_called_once()
        assert send.call_args.args[1] == signal.SIGKILL
        numeric_kill.assert_not_called()
    assert child.wait(timeout=2) == -signal.SIGKILL


async def test_close_does_not_kill_registered_process(orch, processes):
    child = processes()
    orch.process_guards.register(child.pid)
    await orch.process_guards.close()
    assert child.poll() is None
    assert orch.process_guards.snapshots() == []


async def test_duplicate_protected_and_missing_processes(orch, processes, monkeypatch):
    child = processes()
    orch.process_guards.register(child.pid)
    with pytest.raises(ProcessGuardError) as duplicate:
        orch.process_guards.register(child.pid)
    assert duplicate.value.status == 409
    for pid in (os.getpid(), os.getppid(), 1, 0, -1):
        with pytest.raises(ProcessGuardError) as protected:
            orch.process_guards.register(pid)
        assert protected.value.status == 400
    with pytest.raises(ProcessGuardError) as missing:
        orch.process_guards.register(99999999)
    assert missing.value.status == 404
    await orch.process_guards.unregister(child.pid)
    loader = fake_running(orch, monkeypatch)
    loader._process.pid = child.pid
    with pytest.raises(ProcessGuardError, match="router-managed"):
        orch.process_guards.register(child.pid)
    with pytest.raises(ProcessGuardError) as unregistered:
        await orch.process_guards.kill(child.pid)
    assert unregistered.value.status == 404


@pytest.fixture
async def api():
    app = create_app(RouterConfig(
        services=[ServiceConfig(name="model", binary="unused")],
        llama_backends=["model"],
    ))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://router"
    ) as client:
        yield app, client
    await app.state.orch.process_guards.close()


async def test_api_lifecycle_and_status(api, processes):
    app, client = api
    child = processes()
    response = await client.post("/v1/process-guards", json={"pid": child.pid})
    assert response.status_code == 201
    assert response.json()["blocking"]
    duplicate = await client.post("/v1/process-guards", json={"pid": child.pid})
    assert duplicate.status_code == 409
    listed = (await client.get("/v1/process-guards")).json()
    assert listed["requests_blocked"]
    status = (await client.get("/status")).json()
    assert status["requests_blocked"]
    assert status["process_guards"][0]["pid"] == child.pid
    assert (await client.post("/v1/models/unload-all")).json() == {"unloaded": []}
    removed = await client.delete(f"/v1/process-guards/{child.pid}")
    assert removed.status_code == 200
    assert child.poll() is None
    assert (await client.post(f"/v1/process-guards/{child.pid}/kill")).status_code == 404
    await client.post("/v1/process-guards", json={"pid": child.pid, "timeout": 30})
    killed = await client.post(f"/v1/process-guards/{child.pid}/kill")
    assert killed.status_code == 200 and killed.json()["killed"]
    assert (await client.get("/v1/process-guards")).json()["process_guards"] == []


@pytest.mark.parametrize("body", [
    {}, [], {"pid": True}, {"pid": "123"}, {"pid": 1}, {"pid": -10},
    {"pid": 123, "timeout": 0}, {"pid": 123, "timeout": -1},
    {"pid": 123, "timeout": "1"}, {"pid": 123, "timeout": True},
    {"pid": 123, "timeout": "NaN"}, {"pid": 123, "unexpected": 1},
])
async def test_invalid_api_registration(api, body):
    _, client = api
    response = await client.post("/v1/process-guards", json=body)
    assert response.status_code == 422


async def test_api_guard_blocks_chat_but_not_status(api, processes, monkeypatch):
    app, client = api
    child = processes()
    await client.post("/v1/process-guards", json={"pid": child.pid})
    ensure = AsyncMock(side_effect=RuntimeError("mock backend"))
    monkeypatch.setattr(app.state.orch, "ensure_running", ensure)
    request = asyncio.create_task(client.post("/v1/chat/completions", json={
        "model": "model", "messages": [{"role": "user", "content": "test"}],
    }))
    await asyncio.sleep(0.03)
    ensure.assert_not_awaited()
    assert (await client.get("/status")).status_code == 200
    await client.delete(f"/v1/process-guards/{child.pid}")
    assert (await asyncio.wait_for(request, 2)).status_code == 503
    assert app.state.orch.busy_backends() == []


async def test_cancel_proxy_setup_releases_slot(api, monkeypatch):
    app, client = api
    entered = asyncio.Event()

    async def ensure(*args):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(app.state.orch, "ensure_running", ensure)
    pending = asyncio.create_task(client.post("/v1/chat/completions", json={"model": "model"}))
    await asyncio.wait_for(entered.wait(), 1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert app.state.orch.busy_backends() == []
    assert await asyncio.wait_for(app.state.orch.unload_all(), 1) == []


async def test_unload_stale_but_live_backend_handle(orch, monkeypatch):
    loader = fake_running(orch, monkeypatch)
    loader._state = ServiceState.DEAD

    async def force_kill():
        loader._process.returncode = -signal.SIGKILL
        return True

    monkeypatch.setattr(loader, "_force_kill", AsyncMock(side_effect=force_kill))
    assert await orch.unload_all() == ["model"]
    loader._force_kill.assert_awaited_once()
    loader.stop.assert_not_awaited()
    assert orch._extra_vram == {}
