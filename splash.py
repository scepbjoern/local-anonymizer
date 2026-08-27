"""Lightweight splash screen for instant visual feedback on app launch."""
import sys
import tkinter as tk

try:
    root = tk.Tk()
    root.title("Privacy-First Local Anonymizer")
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

    # Auto-destroy after 7 seconds (when app window is loaded)
    root.after(7000, root.destroy)
    root.mainloop()
except Exception:
    pass
