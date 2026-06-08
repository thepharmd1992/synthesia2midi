from PySide6.QtGui import QCloseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

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


def test_dialog_uses_refresh_info_label_and_default_1080p_quality(tmp_path):
    QApplication.instance() or QApplication([])
    dialog = YouTubeDownloadDialog(default_output_dir=str(tmp_path))

    assert dialog.fetch_info_btn.text() == "Refresh Info"
    assert dialog.quality_combo.currentData() == "1080p"
    assert [dialog.quality_combo.itemData(i) for i in range(dialog.quality_combo.count())] == [
        "1080p",
        "720p",
        "480p",
    ]
    assert "faster processing" in dialog.quality_combo.itemText(1)


def test_video_info_success_enables_quality_selector(tmp_path):
    QApplication.instance() or QApplication([])
    dialog = YouTubeDownloadDialog(default_output_dir=str(tmp_path))
    dialog.url_input.setText("https://www.youtube.com/watch?v=SFFSZQCnU_M")

    dialog._on_video_info_fetched(
        "https://www.youtube.com/watch?v=SFFSZQCnU_M",
        {"title": "Mary", "duration": 24, "uploader": "Tuttopiano"},
    )

    assert dialog.quality_combo.isEnabled()


def test_download_starts_with_indeterminate_progress(monkeypatch, tmp_path):
    QApplication.instance() or QApplication([])
    dialog = YouTubeDownloadDialog(default_output_dir=str(tmp_path))
    started = []

    class FakeSignal:
        def connect(self, slot):
            pass

    class FakeDownloadThread:
        def __init__(self, url, output_dir, quality="1080p", overwrite=False):
            self.url = url
            self.output_dir = output_dir
            self.quality = quality
            self.overwrite = overwrite
            self.progress_handler = type(
                "Progress",
                (),
                {
                    "progress": FakeSignal(),
                    "status": FakeSignal(),
                    "finished": FakeSignal(),
                    "error": FakeSignal(),
                },
            )()

        def start(self):
            started.append((self.url, self.quality, self.overwrite))

    monkeypatch.setattr(youtube_download_dialog, "YouTubeDownloaderThread", FakeDownloadThread)

    dialog.url_input.setText("https://www.youtube.com/watch?v=SFFSZQCnU_M")
    dialog._current_info_url = "https://www.youtube.com/watch?v=SFFSZQCnU_M"
    dialog.download_btn.setEnabled(True)
    dialog.start_download()

    assert started == [("https://www.youtube.com/watch?v=SFFSZQCnU_M", "1080p", False)]
    assert dialog.status_label.text() == "Starting download..."
    assert not dialog.progress_bar.isHidden()
    assert dialog.progress_bar.minimum() == 0
    assert dialog.progress_bar.maximum() == 0


def test_download_stall_timer_shows_waiting_status(monkeypatch, tmp_path):
    QApplication.instance() or QApplication([])
    monkeypatch.setattr(YouTubeDownloadDialog, "DOWNLOAD_STALL_DELAY_MS", 1)
    dialog = YouTubeDownloadDialog(default_output_dir=str(tmp_path))

    class FakeSignal:
        def connect(self, slot):
            pass

    class FakeDownloadThread:
        def __init__(self, *args, **kwargs):
            self.progress_handler = type(
                "Progress",
                (),
                {
                    "progress": FakeSignal(),
                    "status": FakeSignal(),
                    "finished": FakeSignal(),
                    "error": FakeSignal(),
                },
            )()

        def start(self):
            pass

        def isRunning(self):
            return True

    monkeypatch.setattr(youtube_download_dialog, "YouTubeDownloaderThread", FakeDownloadThread)

    dialog.url_input.setText("https://www.youtube.com/watch?v=SFFSZQCnU_M")
    dialog._current_info_url = "https://www.youtube.com/watch?v=SFFSZQCnU_M"
    dialog.download_btn.setEnabled(True)
    dialog.start_download()

    QTest.qWait(20)

    assert dialog.status_label.text() == "Still waiting for YouTube..."


def test_selected_quality_is_passed_to_download_thread(monkeypatch, tmp_path):
    QApplication.instance() or QApplication([])
    dialog = YouTubeDownloadDialog(default_output_dir=str(tmp_path))
    started = []

    class FakeSignal:
        def connect(self, slot):
            pass

    class FakeDownloadThread:
        def __init__(self, url, output_dir, quality="1080p", overwrite=False):
            self.progress_handler = type(
                "Progress",
                (),
                {
                    "progress": FakeSignal(),
                    "status": FakeSignal(),
                    "finished": FakeSignal(),
                    "error": FakeSignal(),
                },
            )()
            self.quality = quality

        def start(self):
            started.append(self.quality)

    monkeypatch.setattr(youtube_download_dialog, "YouTubeDownloaderThread", FakeDownloadThread)

    dialog.url_input.setText("https://www.youtube.com/watch?v=SFFSZQCnU_M")
    dialog._current_info_url = "https://www.youtube.com/watch?v=SFFSZQCnU_M"
    dialog.quality_combo.setCurrentIndex(dialog.quality_combo.findData("480p"))
    dialog.download_btn.setEnabled(True)
    dialog.start_download()

    assert started == ["480p"]


def test_existing_download_can_be_reused_without_starting_download(monkeypatch, tmp_path):
    QApplication.instance() or QApplication([])
    dialog = YouTubeDownloadDialog(default_output_dir=str(tmp_path))
    emitted = []

    existing = tmp_path / "mary" / "mary_1080p.mp4"
    existing.parent.mkdir()
    existing.write_text("video")
    dialog.video_downloaded.connect(emitted.append)
    dialog.url_input.setText("https://www.youtube.com/watch?v=SFFSZQCnU_M")
    dialog._current_info_url = "https://www.youtube.com/watch?v=SFFSZQCnU_M"
    dialog._current_video_title = "Mary"
    dialog.download_btn.setEnabled(True)

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.Yes,
    )

    dialog.start_download()

    assert emitted == [str(existing)]
    assert dialog.result() == QDialog.Accepted
