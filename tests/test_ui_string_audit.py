from pathlib import Path


def test_static_extractor_finds_qt_visible_strings_and_classifies_literals(tmp_path):
    from synthesia2midi.tools.audit_ui_strings import collect_static_candidates

    source = tmp_path / "sample_ui.py"
    source.write_text(
        "\n".join(
            [
                "import logging",
                "from PySide6.QtWidgets import QLabel, QMessageBox, QComboBox",
                "class Dialog:",
                "    def setup_ui(self):",
                "        label = QLabel('Ready to convert')",
                "        self.url_input.setPlaceholderText('https://www.youtube.com/watch?v=...')",
                "        self.browser_combo = QComboBox()",
                "        self.browser_combo.addItem('Chrome', 'chrome')",
                "        QMessageBox.warning(self, 'Download Error', f'Failed to download video: {error}')",
                "        logging.warning('log-only message')",
            ]
        ),
        encoding="utf-8",
    )

    candidates = collect_static_candidates([source], root=tmp_path)
    by_text = {candidate.text: candidate for candidate in candidates}

    assert by_text["Ready to convert"].classification == "translate"
    assert by_text["Download Error"].classification == "translate"
    assert by_text["Failed to download video: {...}"].classification == "translate"
    assert by_text["https://www.youtube.com/watch?v=..."].classification == "path_or_url"
    assert by_text["Chrome"].classification == "do_not_translate"
    assert by_text["chrome"].classification == "technical_id"
    assert "log-only message" not in by_text


def test_runtime_widget_crawler_collects_visible_text():
    from PySide6.QtWidgets import QApplication

    from synthesia2midi.gui.startup_dialog import StartupDialog
    from synthesia2midi.tools.audit_ui_strings import collect_widget_text

    app = QApplication.instance() or QApplication([])
    dialog = StartupDialog(recent_video_paths=["/tmp/example.mp4"])

    try:
        candidates = collect_widget_text(dialog)
        by_text = {candidate.text: candidate for candidate in candidates}

        assert by_text["Open Video File"].origin == "runtime"
        assert by_text["Recent Videos"].classification == "translate"
        assert by_text["example.mp4"].classification == "dynamic_user_data"
        assert by_text["/tmp/example.mp4"].classification == "dynamic_user_data"
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_audit_manifest_has_stable_json_shape(tmp_path):
    from synthesia2midi.tools.audit_ui_strings import UiStringCandidate, write_manifest

    output = tmp_path / "manifest.json"
    write_manifest(
        [
            UiStringCandidate(
                text="Open Video File",
                classification="translate",
                origin="static",
                source="sample.py",
                line=10,
                context="QPushButton",
                role="arg0",
            )
        ],
        output,
    )

    manifest = output.read_text(encoding="utf-8")

    assert '"schema_version": 1' in manifest
    assert '"text": "Open Video File"' in manifest
    assert '"classification": "translate"' in manifest
