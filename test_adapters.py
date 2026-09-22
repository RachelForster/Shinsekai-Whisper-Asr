from __future__ import annotations

import threading

from plugins.whisper_asr.adapters import RealtimeSTTAdapter


class _Recorder:
    def __init__(self) -> None:
        self.aborted = threading.Event()
        self.listen_count = 0
        self.shutdown_count = 0

    def abort(self) -> None:
        self.aborted.set()

    def listen(self) -> None:
        self.listen_count += 1

    def shutdown(self) -> None:
        self.shutdown_count += 1


def test_finish_hold_keeps_realtime_recorder_warm() -> None:
    adapter = RealtimeSTTAdapter("zh", lambda _text, _partial: None)
    recorder = _Recorder()
    adapter._recorder = recorder
    adapter._is_running = True

    assert adapter.finish_hold() is True
    assert recorder.aborted.wait(timeout=1)
    assert adapter._recorder is recorder
    assert recorder.shutdown_count == 0

    adapter.resume()

    assert adapter._recorder is recorder
    assert recorder.listen_count == 1
