"""Lightweight splash screen for instant visual feedback on app launch."""
import sys
import tkinter as tk
from pathlib import Path

CONFIG_DIR = Path.home() / ".local-anonymizer"
READY_FLAG = CONFIG_DIR / "splash_ready.tmp"

# Clear any leftover flag
try:
    READY_FLAG.unlink(missing_ok=True)
except Exception:
    pass

try:
    root = tk.Tk()
    root.title("Privacy-First Local Anonymizer Splash")
    root.geometry("420x150")
    root.overrideredirect(True)
    root.eval("tk::PlaceWindow . center")
    root.configure(bg="#0f172a")

    frame = tk.Frame(root, bg="#0f172a", padx=24, pady=24, highlightbackground="#38bdf8", highlightthickness=1)
    frame.pack(expand=True, fill="both")

    title = tk.Label(frame, text="🔒 Privacy-First Local Anonymizer", font=("Segoe UI", 13, "bold"), fg="#38bdf8", bg="#0f172a")
    title.pack(pady=(0, 8))

    status = tk.Label(frame, text="Applikation wird geladen... Bitte einen Moment Geduld.", font=("Segoe UI", 9), fg="#e2e8f0", bg="#0f172a")
    status.pack()

    badge = tk.Label(frame, text="100% Lokal & Offline • Keine Cloud", font=("Segoe UI", 8, "italic"), fg="#94a3b8", bg="#0f172a")
    badge.pack(pady=(10, 0))

    def bring_main_to_front():
        if sys.platform == "win32":
            try:
                import ctypes
                user32 = ctypes.windll.user32
                hwnd = user32.FindWindowW(None, "Privacy-First Local Anonymizer")
                if hwnd:
                    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                    user32.SetForegroundWindow(hwnd)
                    user32.BringWindowToTop(hwnd)
            except Exception:
                pass

    def check_ready():
        if READY_FLAG.exists():
            try:
                READY_FLAG.unlink(missing_ok=True)
            except Exception:
                pass
            bring_main_to_front()
            root.destroy()
        else:
            root.after(100, check_ready)

    # Check every 100ms if main app is ready; max fallback timeout 25s
    root.after(100, check_ready)
    root.after(25000, root.destroy)
    root.mainloop()
except Exception:
    pass
