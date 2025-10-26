import tkinter as tk
from tkinter import ttk

from tanks_widget import TanksWidget
from quadcopter_widget import QuadcopterWidget


class VisualWindow:
    """
    A modeless popup window that hosts either the TanksWidget or QuadcopterWidget.
    The window can be shown/hidden and safely recreated on problem changes.
    """
    def __init__(self, root, title="Visual"):
        self.root = root
        self.win = tk.Toplevel(root)
        self.win.title(title)
        self.win.withdraw()  # start hidden; we’ll show it on app init
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # Let the popup be resized independently
        self.win.rowconfigure(0, weight=1)
        self.win.columnconfigure(0, weight=1)

        self.container = ttk.Frame(self.win, padding=6)
        self.container.grid(row=0, column=0, sticky="nsew")

        self.visual_widget = None
        self.kind = None  # "tanks" | "quad"

    def show(self):
        self.win.deiconify()
        self.win.lift()

    def hide(self):
        self.win.withdraw()

    def _on_close(self):
        # Don't destroy: just hide so we can re-open quickly.
        self.hide()

    def set_kind(self, kind: str, n: int, title: str):
        """
        kind: "tanks" or "quad"
        n:    number of tanks/rotors
        """
        # Clean out old content
        for ch in list(self.container.winfo_children()):
            try:
                ch.destroy()
            except Exception:
                pass

        if kind == "quad":
            self.visual_widget = QuadcopterWidget(self.container, n_rotors=n, title=title)
            self.win.title("Quadcopter")
        else:
            self.visual_widget = TanksWidget(self.container, n_tanks=n, title=title)
            self.win.title("5-Tank Water System")

        self.visual_widget.pack(fill="both", expand=True)
        self.kind = kind

    # Unified pass-through used by App.loop()
    def push_state(self, **kwargs):
        if self.visual_widget and self.visual_widget.winfo_exists():
            try:
                self.visual_widget.push_state(**kwargs)
            except Exception:
                # Never let the visual kill the main loop
                pass
