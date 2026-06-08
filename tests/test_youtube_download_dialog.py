from PySide6.QtGui import QCloseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from synthesia2midi.gui import youtube_download_dialog
from synthesia2midi.gui.youtube_download_dialog import YouTubeDownloadDialog


def test_valid_url_auto_fetches_after_debounce(monkeypatch, tmp_path):
    QApplication.instance() or QApplication([])
    monkeypatch.setattr(YouTubeDownloadDialog, "AUTO_FETCH_DELAY_MS", 1)
    dialog = YouTubeDownloadDialog(default_output_dir=str(tmp_path))
    calls = []

    def record_fetch():
        calls.append(dialog.url_input.text())

    monkeypatch.setattr(dialog, "fetch_video_info", record_fetch)

    dialog.url_input.setText("https://www.youtube.com/watch?v=SFFSZQCnU_M")

    QTest.qWait(20)

    assert calls == ["https://www.youtube.com/watch?v=SFFSZQCnU_M"]


def test_url_change_clears_stale_video_info_and_disables_download(tmp_path):
    QApplication.instance() or QApplication([])
    dialog = YouTubeDownloadDialog(default_output_dir=str(tmp_path))
    dialog.show()
    QTest.qWait(1)
    dialog.info_widget.show()
    dialog.download_btn.setEnabled(True)

    dialog.url_input.setText("https://www.youtube.com/watch?v=SFFSZQCnU_M")

    assert dialog.info_widget.isHidden()
    assert not dialog.download_btn.isEnabled()


def test_fetch_video_info_starts_background_worker(monkeypatch, tmp_path):
    QApplication.instance() or QApplication([])
    dialog = YouTubeDownloadDialog(default_output_dir=str(tmp_path))
    started = []
    warnings = []

    class FakeSignal:
        def connect(self, slot):
            pass

    class FakeInfoThread:
        info_fetched = FakeSignal()
        error = FakeSignal()

        def __init__(self, url, output_dir):
            self.url = url
            self.output_dir = output_dir

        def isRunning(self):
            return False

        def start(self):
            started.append((self.url, self.output_dir))

    def fail_inline_fetch(url):
        raise AssertionError("fetch_video_info should not fetch inline")

    monkeypatch.setattr(dialog.downloader, "get_video_info", fail_inline_fetch)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: warnings.append(args))
    monkeypatch.setattr(youtube_download_dialog, "YouTubeInfoFetcherThread", FakeInfoThread, raising=False)

    dialog.url_input.setText("https://www.youtube.com/watch?v=SFFSZQCnU_M")
    dialog.auto_fetch_timer.stop()
    dialog.fetch_video_info()

    assert started == [("https://www.youtube.com/watch?v=SFFSZQCnU_M", str(tmp_path))]
    assert warnings == []


def test_close_waits_for_in_progress_video_info_fetch(tmp_path):
    QApplication.instance() or QApplication([])
    dialog = YouTubeDownloadDialog(default_output_dir=str(tmp_path))
    calls = []

    class FakeInfoThread:
        def isRunning(self):
            return True

        def quit(self):
            calls.append("quit")

        def wait(self):
            calls.append("wait")

    dialog.info_fetch_thread = FakeInfoThread()

    dialog.closeEvent(QCloseEvent())

    assert calls == ["quit", "wait"]
