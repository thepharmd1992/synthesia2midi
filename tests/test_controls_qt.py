from PySide6.QtWidgets import QApplication

from synthesia2midi.gui.controls_qt import ControlPanelQt


def test_edit_midi_button_emits_touchup_request_directly():
    QApplication.instance() or QApplication([])
    panel = ControlPanelQt()
    emitted = []

    panel.midi_touchup_requested.connect(lambda: emitted.append(True))
    panel.midi_touchup_button.click()

    assert emitted == [True]
