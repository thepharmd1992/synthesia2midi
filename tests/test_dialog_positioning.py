from types import SimpleNamespace

from synthesia2midi.gui.dialog_positioning import screen_for_widget


def test_screen_resolution_prefers_parent_then_dialog(monkeypatch):
    parent_screen = object()
    dialog_screen = object()
    primary_screen = object()
    monkeypatch.setattr(
        "synthesia2midi.gui.dialog_positioning.QApplication.primaryScreen",
        lambda: primary_screen,
    )

    assert (
        screen_for_widget(
            SimpleNamespace(screen=lambda: parent_screen),
            SimpleNamespace(screen=lambda: dialog_screen),
        )
        is parent_screen
    )
    assert (
        screen_for_widget(None, SimpleNamespace(screen=lambda: dialog_screen))
        is dialog_screen
    )
    assert screen_for_widget(None, None) is primary_screen
