from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QBoxLayout, QDialog, QMessageBox

from synthesia2midi.gui import youtube_download_dialog
from synthesia2midi.gui.youtube_download_dialog import YouTubeDownloadDialog


class FakeSettings:
    def __init__(self, initial=None):
        self.values = dict(initial or {})

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value


def _wait_until(predicate, *, timeout_ms=500):
    deadline = timeout_ms
    while deadline >= 0:
        QApplication.processEvents()
        if predicate():
            return True
        QTest.qWait(10)
        deadline -= 10
    return predicate()


def test_download_dialog_stacks_actions_and_wraps_metadata():
    QApplication.instance() or QApplication([])
    dialog = YouTubeDownloadDialog(default_output_dir="/tmp")
    try:
        assert dialog.action_layout.direction() == QBoxLayout.TopToBottom
        assert dialog.title_label.wordWrap()
        assert dialog.duration_label.wordWrap()
        assert dialog.uploader_label.wordWrap()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_dialog_preserves_default_download_dir(tmp_path):
    QApplication.instance() or QApplication([])
    download_dir = tmp_path / "Downloads" / "Synthesia2MIDI"

    dialog = YouTubeDownloadDialog(default_output_dir=str(download_dir))

    assert dialog.downloader.output_dir == download_dir


def test_valid_url_auto_fetches_after_debounce(monkeypatch, tmp_path):
    QApplication.instance() or QApplication([])
    monkeypatch.setattr(YouTubeDownloadDialog, "AUTO_FETCH_DELAY_MS", 1)
    dialog = YouTubeDownloadDialog(default_output_dir=str(tmp_path))
    calls = []

    def record_fetch():
        calls.append(dialog.url_input.text())

    monkeypatch.setattr(dialog, "fetch_video_info", record_fetch)

    dialog.url_input.setText("https://www.youtube.com/watch?v=SFFSZQCnU_M")

    assert _wait_until(lambda: len(calls) == 1, timeout_ms=500)

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
    assert dialog.quality_combo.itemText(0) == "1080p - recommended for best MIDI detection"
    assert dialog.quality_combo.itemText(1) == "720p - faster, may be less accurate"
    assert dialog.quality_combo.itemText(2) == "480p - fastest, highest risk of bad calibration"
    assert [dialog.quality_combo.itemData(i) for i in range(dialog.quality_combo.count())] == [
        "1080p",
        "720p",
        "480p",
    ]
    assert dialog.browser_combo.currentData() == "chrome"
    assert dialog.auto_retry_checkbox.isChecked()
    assert dialog.fallback_group.title() == "If YouTube blocks the download"
    assert dialog.fallback_group.isHidden()
    assert dialog.fallback_hint_label.text() == (
        "Synthesia2MIDI can retry using saved browser cookies only if YouTube blocks the normal download."
    )


def test_cookie_fallback_appears_only_when_downloader_reports_cookie_retry(tmp_path):
    QApplication.instance() or QApplication([])
    dialog = YouTubeDownloadDialog(default_output_dir=str(tmp_path))
    dialog.show()
    QApplication.processEvents()

    assert dialog.fallback_group.isHidden()

    dialog.update_status(
        "Video info request failed. Retrying with Safari browser cookies..."
    )
    QApplication.processEvents()

    assert dialog.fallback_group.isVisible()


def test_dialog_restores_saved_cookie_retry_preferences(tmp_path):
    QApplication.instance() or QApplication([])
    settings = FakeSettings(
        {
            youtube_download_dialog.YOUTUBE_PREFERRED_BROWSER_KEY: "safari",
            youtube_download_dialog.YOUTUBE_AUTO_COOKIE_RETRY_KEY: False,
        }
    )

    dialog = YouTubeDownloadDialog(default_output_dir=str(tmp_path), settings=settings)

    assert dialog.browser_combo.currentData() == "safari"
    assert not dialog.auto_retry_checkbox.isChecked()


def test_dialog_persists_cookie_retry_preferences(tmp_path):
    QApplication.instance() or QApplication([])
    settings = FakeSettings()
    dialog = YouTubeDownloadDialog(default_output_dir=str(tmp_path), settings=settings)

    dialog.browser_combo.setCurrentIndex(dialog.browser_combo.findData("edge"))
    dialog.auto_retry_checkbox.setChecked(False)

    assert settings.values[youtube_download_dialog.YOUTUBE_PREFERRED_BROWSER_KEY] == "edge"
    assert settings.values[youtube_download_dialog.YOUTUBE_AUTO_COOKIE_RETRY_KEY] is False


def test_video_info_success_enables_quality_selector(tmp_path):
    QApplication.instance() or QApplication([])
    dialog = YouTubeDownloadDialog(default_output_dir=str(tmp_path))
    dialog.url_input.setText("https://www.youtube.com/watch?v=SFFSZQCnU_M")

    dialog._on_video_info_fetched(
        "https://www.youtube.com/watch?v=SFFSZQCnU_M",
        {"title": "Mary", "duration": 24, "uploader": "Tuttopiano"},
    )

    assert dialog.quality_combo.isEnabled()


def test_video_info_group_expands_to_fit_metadata_labels(tmp_path):
    QApplication.instance() or QApplication([])
    dialog = YouTubeDownloadDialog(default_output_dir=str(tmp_path))
    dialog.show()
    dialog.url_input.setText("https://www.youtube.com/watch?v=SFFSZQCnU_M")
    dialog.auto_fetch_timer.stop()

    dialog._on_video_info_fetched(
        "https://www.youtube.com/watch?v=SFFSZQCnU_M",
        {
            "title": "Nursery Rhymes - Mary had a little lamb (Piano for children)",
            "duration": 24,
            "uploader": "Tuttopiano",
        },
    )
    QApplication.processEvents()

    assert dialog.info_widget.height() >= dialog.info_widget.sizeHint().height()


def test_video_info_group_does_not_overlap_quality_selector_with_wide_font(tmp_path):
    app = QApplication.instance() or QApplication([])
    original_font = QFont(app.font())
    wide_font = QFont(original_font)
    wide_font.setPointSizeF(original_font.pointSizeF() * 1.5)
    wide_font.setStretch(135)
    app.setFont(wide_font)
    dialog = YouTubeDownloadDialog(default_output_dir=str(tmp_path))
    try:
        dialog.show()
        dialog.url_input.setText("https://www.youtube.com/watch?v=SFFSZQCnU_M")
        dialog.auto_fetch_timer.stop()
        dialog._on_video_info_fetched(
            "https://www.youtube.com/watch?v=SFFSZQCnU_M",
            {
                "title": "A Long but Fully Visible Synthesia Piano Tutorial Title",
                "duration": 754,
                "uploader": "Example Piano Channel",
            },
        )
        QApplication.processEvents()

        assert dialog.info_widget.geometry().bottom() < dialog.quality_combo.geometry().top()
    finally:
        dialog.close()
        dialog.deleteLater()
        app.setFont(original_font)


def test_video_info_success_uses_real_available_quality_options(tmp_path):
    QApplication.instance() or QApplication([])
    dialog = YouTubeDownloadDialog(default_output_dir=str(tmp_path))
    dialog.url_input.setText("https://www.youtube.com/watch?v=SFFSZQCnU_M")

    dialog._on_video_info_fetched(
        "https://www.youtube.com/watch?v=SFFSZQCnU_M",
        {
            "title": "Mary",
            "duration": 24,
            "uploader": "Tuttopiano",
            "available_qualities": {
                "1080p": {
                    "available": True,
                    "actual_height": 720,
                    "note": "recommended for best MIDI detection",
                },
                "720p": {
                    "available": True,
                    "actual_height": 720,
                    "note": "faster, may be less accurate",
                },
                "480p": {
                    "available": True,
                    "actual_height": 360,
                    "note": "fastest, highest risk of bad calibration",
                },
            },
        },
    )

    assert [dialog.quality_combo.itemData(i) for i in range(dialog.quality_combo.count())] == [
        "720p",
        "480p",
    ]
    assert dialog.quality_combo.currentData() == "720p"
    assert dialog.quality_combo.itemText(1).startswith("Up to 480p (360p source)")


def test_video_info_success_rewords_legacy_quality_notes(tmp_path):
    QApplication.instance() or QApplication([])
    dialog = YouTubeDownloadDialog(default_output_dir=str(tmp_path))
    dialog.url_input.setText("https://www.youtube.com/watch?v=SFFSZQCnU_M")

    dialog._on_video_info_fetched(
        "https://www.youtube.com/watch?v=SFFSZQCnU_M",
        {
            "title": "Mary",
            "duration": 24,
            "uploader": "Tuttopiano",
            "available_qualities": {
                "1080p": {"available": True, "actual_height": 1080, "note": "Highest detail"},
                "720p": {
                    "available": True,
                    "actual_height": 720,
                    "note": "Faster processing, higher calibration risk",
                },
                "480p": {
                    "available": True,
                    "actual_height": 360,
                    "note": "Fastest processing, highest calibration risk",
                },
            },
        },
    )

    assert dialog.quality_combo.itemText(0) == (
        "Up to 1080p (1080p source) - recommended for best MIDI detection"
    )
    assert dialog.quality_combo.itemText(2) == (
        "Up to 480p (360p source) - fastest, highest risk of bad calibration"
    )


def test_download_starts_with_indeterminate_progress(monkeypatch, tmp_path):
    QApplication.instance() or QApplication([])
    settings = FakeSettings({youtube_download_dialog.YOUTUBE_PREFERRED_BROWSER_KEY: "safari"})
    dialog = YouTubeDownloadDialog(default_output_dir=str(tmp_path), settings=settings)
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
            started.append(
                (
                    self.url,
                    self.quality,
                    self.overwrite,
                    getattr(self, "preferred_browser", None),
                    getattr(self, "auto_cookie_retry", None),
                )
            )

    monkeypatch.setattr(youtube_download_dialog, "YouTubeDownloaderThread", FakeDownloadThread)

    dialog.url_input.setText("https://www.youtube.com/watch?v=SFFSZQCnU_M")
    dialog._current_info_url = "https://www.youtube.com/watch?v=SFFSZQCnU_M"
    dialog.download_btn.setEnabled(True)
    dialog.start_download()

    assert started == [
        ("https://www.youtube.com/watch?v=SFFSZQCnU_M", "1080p", False, "safari", True)
    ]
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

    assert _wait_until(
        lambda: dialog.status_label.text() == "Still waiting for YouTube...",
        timeout_ms=500,
    )

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


def test_fetch_video_info_passes_cookie_retry_preferences_to_worker(monkeypatch, tmp_path):
    QApplication.instance() or QApplication([])
    settings = FakeSettings(
        {
            youtube_download_dialog.YOUTUBE_PREFERRED_BROWSER_KEY: "edge",
            youtube_download_dialog.YOUTUBE_AUTO_COOKIE_RETRY_KEY: True,
        }
    )
    dialog = YouTubeDownloadDialog(default_output_dir=str(tmp_path), settings=settings)
    started = []

    class FakeSignal:
        def connect(self, slot):
            pass

    class FakeInfoThread:
        info_fetched = FakeSignal()
        error = FakeSignal()
        finished = FakeSignal()

        def __init__(self, url, output_dir):
            self.url = url
            self.output_dir = output_dir

        def isRunning(self):
            return False

        def start(self):
            started.append(
                (
                    self.url,
                    self.output_dir,
                    getattr(self, "preferred_browser", None),
                    getattr(self, "auto_cookie_retry", None),
                )
            )

    monkeypatch.setattr(youtube_download_dialog, "YouTubeInfoFetcherThread", FakeInfoThread, raising=False)

    dialog.url_input.setText("https://www.youtube.com/watch?v=SFFSZQCnU_M")
    dialog.auto_fetch_timer.stop()
    dialog.fetch_video_info()

    assert started == [("https://www.youtube.com/watch?v=SFFSZQCnU_M", str(tmp_path), "edge", True)]


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
