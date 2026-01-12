"""
Utility functions for managing spinbox widgets and preventing accidental wheel scrolling.
"""
# Standard library imports
from typing import Union

# Third-party imports
from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox, QWidget


class SpinBoxWheelFilter(QObject):
    """Event filter to prevent wheel events on unfocused spinboxes."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.unfocus_timers = {}  # Track timers for each spinbox
        
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Block wheel events on unfocused spinboxes and provide auto-unfocus."""
        if isinstance(obj, (QSpinBox, QDoubleSpinBox)):
            widget_name = obj.objectName() or f"{obj.__class__.__name__}@{id(obj):x}"
            
            if event.type() == QEvent.Wheel:
                # Only allow wheel events on focused spinboxes
                if not obj.hasFocus():
                    return True  # Block the event
                    
            elif event.type() == QEvent.FocusIn:
                # Cancel any pending unfocus timer
                if widget_name in self.unfocus_timers:
                    self.unfocus_timers[widget_name].stop()
                    del self.unfocus_timers[widget_name]
                
            elif event.type() == QEvent.FocusOut:
                # Clean up timer if it exists
                if widget_name in self.unfocus_timers:
                    self.unfocus_timers[widget_name].stop()
                    del self.unfocus_timers[widget_name]
                    
            elif event.type() == QEvent.Leave:
                # Mouse left the spinbox area - schedule auto-unfocus
                if obj.hasFocus() and widget_name not in self.unfocus_timers:
                    timer = QTimer()
                    timer.setSingleShot(True)
                    timer.timeout.connect(lambda: self._auto_unfocus_spinbox(obj, widget_name))
                    timer.start(500)  # 500ms delay
                    self.unfocus_timers[widget_name] = timer
                        
            elif event.type() == QEvent.Enter:
                # Mouse entered the spinbox area - cancel auto-unfocus
                if widget_name in self.unfocus_timers:
                    self.unfocus_timers[widget_name].stop()
                    del self.unfocus_timers[widget_name]
                
        return super().eventFilter(obj, event)
    
    def _auto_unfocus_spinbox(self, spinbox, widget_name):
        """Auto-unfocus a spinbox after mouse leaves."""
        if spinbox.hasFocus():
            spinbox.clearFocus()
        # Clean up the timer
        if widget_name in self.unfocus_timers:
            del self.unfocus_timers[widget_name]


def install_spinbox_wheel_filter(widget: QWidget) -> None:
    """
    Install wheel event filter on all spinboxes within a widget to prevent accidental changes.
    
    Args:
        widget: The parent widget to search for spinboxes
    """
    # Create a single filter instance that will be shared by all spinboxes
    wheel_filter = SpinBoxWheelFilter(widget)
    
    # Find QSpinBox and QDoubleSpinBox widgets separately since findChildren doesn't accept tuples
    spinboxes = widget.findChildren(QSpinBox)
    double_spinboxes = widget.findChildren(QDoubleSpinBox)
    
    # Install the filter on all found spinboxes
    for i, spinbox in enumerate(spinboxes + double_spinboxes):
        # Set object name for easier identification if not already set
        if not spinbox.objectName():
            spinbox.setObjectName(f"{spinbox.__class__.__name__}_{i}")
        
        # Set focus policy to only accept focus from explicit clicks
        # This prevents hover focus which causes unwanted wheel events
        spinbox.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        
        # Install the wheel filter
        spinbox.installEventFilter(wheel_filter)


def create_protected_spinbox(*args, **kwargs) -> QSpinBox:
    """
    Create a QSpinBox with wheel event filtering and click-only focus policy.
    
    Args:
        *args, **kwargs: Arguments passed to QSpinBox constructor
        
    Returns:
        QSpinBox with wheel event filtering and proper focus policy
    """
    spinbox = QSpinBox(*args, **kwargs)
    spinbox.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
    wheel_filter = SpinBoxWheelFilter(spinbox)
    spinbox.installEventFilter(wheel_filter)
    return spinbox


def create_protected_double_spinbox(*args, **kwargs) -> QDoubleSpinBox:
    """
    Create a QDoubleSpinBox with wheel event filtering and click-only focus policy.
    
    Args:
        *args, **kwargs: Arguments passed to QDoubleSpinBox constructor
        
    Returns:
        QDoubleSpinBox with wheel event filtering and proper focus policy
    """
    spinbox = QDoubleSpinBox(*args, **kwargs)
    spinbox.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
    wheel_filter = SpinBoxWheelFilter(spinbox)
    spinbox.installEventFilter(wheel_filter)
    return spinbox