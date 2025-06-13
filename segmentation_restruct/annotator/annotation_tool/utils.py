from qtpy.QtWidgets import QWidget, QLabel, QLayoutItem, QPushButton
from qtpy.QtCore import QTimer
from napari import Viewer
from napari.layers import Points, Labels, Layer
from functools import partial
import re

"""This is mostly a hacky workaround to disable functionality of the
napari ui to stop the user of doing things we don't want them to do.
This could not work in the future if things on the napari implementation
change.
See also here: https://forum.image.sc/t/disable-buttons-on-shapes-layer/55243/3"""


def restrict_brush_layer_tools(
    viewer: Viewer, brush_layer: Labels, points_layer: Points
):
    hide_brush_controls = partial(
        hide_unwanted_layer_controls,
        allowed_labels={"opacity", "brush size", "label"},
    )

    hide_points_controls = partial(
        hide_unwanted_layer_controls,
        allowed_labels={"opacity", "point size"},
    )

    # Schedule it to run after the UI is fully built
    QTimer.singleShot(100, lambda: hide_unwanted_buttons(viewer, brush_layer))
    QTimer.singleShot(200, lambda: hide_brush_controls(viewer, brush_layer))
    QTimer.singleShot(250, lambda: hide_points_controls(viewer, points_layer))
    QTimer.singleShot(300, lambda: hide_layerlist_buttons(viewer))


def hide_unwanted_layer_controls(
    viewer: Viewer, layer: Layer, allowed_labels: set[str]
):
    qt_controls = viewer.window._qt_viewer.controls
    layer_controls = qt_controls.widgets.get(layer)

    if layer_controls is None:
        print(f"[DEBUG] Controls for layer '{layer.name}' not ready.")
        return

    print(f"[DEBUG] Hiding unwanted controls for layer: {layer.name}")
    layout = layer_controls.layout()
    if not layout:
        print(f"[DEBUG] No layout found in layer controls: {layer.name}")
        return

    for i in range(layout.count()):
        item = layout.itemAt(i)
        if not isinstance(item, QLayoutItem):
            continue

        widget = item.widget()
        if widget is None:
            continue

        if isinstance(widget, QLabel):
            label_text = widget.text().strip(":").lower()
            if label_text not in allowed_labels:
                print(
                    f"[DEBUG] Hiding row with label '{label_text}' in layer '{layer.name}'"
                )
                widget.hide()

                next_item = layout.itemAt(i + 1)
                if next_item:
                    next_widget = next_item.widget()
                    if next_widget:
                        next_widget.hide()


# Delay execution until the viewer and all widgets are fully initialized
def hide_unwanted_buttons(viewer: Viewer, brush_layer: Labels):
    qt_controls = viewer.window._qt_viewer.controls
    brush_controls = qt_controls.widgets.get(brush_layer)

    if brush_controls is None:
        print("[DEBUG] Brush controls not ready.")
        return

    print("[DEBUG] Hiding unwanted brush mode buttons...")

    # Look through all children widgets and hide unwanted ones
    for child in brush_controls.findChildren(QWidget):
        if hasattr(child, "toolTip"):
            tip = str(child.toolTip()).lower()
            if any(
                mode in tip
                for mode in [
                    "fill",
                    "erase",
                    "erase label",
                    "bucket",
                    "polygon",
                    "pick",
                    "shuffle",
                ]
            ):
                print(f"[DEBUG] Hiding button with tooltip: {tip}")
                child.hide()


def hide_layerlist_buttons(viewer: Viewer):
    buttons_widget = viewer.window._qt_viewer.layerButtons

    if buttons_widget is None:
        print("[DEBUG] Layer buttons panel not found.")
        return

    print("[DEBUG] Hiding layer control buttons with regex...")
    pattern = re.compile(r"^(new|delete).*layer")

    for btn in buttons_widget.findChildren(QPushButton):
        tooltip = btn.toolTip().strip().lower()
        print(f"[DEBUG] Tooltip: {tooltip}")

        if pattern.match(tooltip):
            print(f"[DEBUG] Hiding button with tooltip: {tooltip}")
            btn.hide()
