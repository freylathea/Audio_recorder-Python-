import os
import time
import threading
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class VoiceRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Recorder")
        self.root.geometry("480x680")
        self.root.configure(bg="#0f172a")

        # Recording state variables
        self.is_recording = False
        self.is_paused = False
        self.elapsed_seconds = 0
        self.recording_count = 0
        self.audio_data = []
        self.sample_rate = 44100

        # UI Color Theme matching original CSS
        self.COLOR_BG = "#0f172a"
        self.COLOR_CARD = "#1e293b"
        self.COLOR_PRIMARY = "#ef4444"
        self.COLOR_BORDER = "#334155"
        self.COLOR_TEXT = "#f8fafc"
        self.COLOR_SUBTEXT = "#94a3b8"

        self._setup_ui()

    def _setup_ui(self):
        # Main container
        self.container = tk.Frame(self.root, bg=self.COLOR_BG)
        self.container.pack(fill="both", expand=True, padx=20, pady=20)

        # Recorder Card
        self.card = tk.Frame(self.container, bg=self.COLOR_CARD, highlightbackground=self.COLOR_BORDER, highlightthickness=1)
        self.card.pack(fill="x", pady=(0, 15))

        # Title
        self.title_label = tk.Label(self.card, text="Voice Recorder", font=("Segoe UI", 16, "bold"), fg=self.COLOR_TEXT, bg=self.COLOR_CARD)
        self.title_label.pack(pady=(15, 2))

        # Status
        self.status_label = tk.Label(self.card, text="Ready to record", font=("Segoe UI", 9), fg=self.COLOR_SUBTEXT, bg=self.COLOR_CARD)
        self.status_label.pack(pady=(0, 10))

        # Timer
        self.timer_label = tk.Label(self.card, text="00:00", font=("Segoe UI", 32, "bold"), fg=self.COLOR_TEXT, bg=self.COLOR_CARD)
        self.timer_label.pack(pady=(0, 10))

        # Canvas Visualizer
        self.fig = Figure(figsize=(4, 0.8), facecolor="#162032")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#162032")
        self.ax.axis("off")
        self.bars = self.ax.bar(range(32), np.zeros(32), color="#ef4444", width=0.8)
        self.fig.tight_layout(pad=0)

        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=self.card)
        self.canvas_widget.get_tk_widget().pack(fill="x", padx=15, pady=(0, 15))

        # Controls Button Frame
        self.controls_frame = tk.Frame(self.card, bg=self.COLOR_CARD)
        self.controls_frame.pack(pady=(0, 15))

        self.btn_record = tk.Button(self.controls_frame, text="⏺ Record", font=("Segoe UI", 10, "bold"), fg=self.COLOR_TEXT, bg=self.COLOR_PRIMARY,
                                    activebackground="#dc2626", activeforeground=self.COLOR_TEXT, bd=0, padx=12, pady=6, command=self.start_recording)
        self.btn_record.pack(side="left", padx=5)

        self.btn_pause = tk.Button(self.controls_frame, text="⏸ Pause", font=("Segoe UI", 10, "bold"), fg=self.COLOR_TEXT, bg=self.COLOR_BORDER,
                                   activebackground="#475569", activeforeground=self.COLOR_TEXT, bd=0, padx=12, pady=6, state="disabled", command=self.toggle_pause)
        self.btn_pause.pack(side="left", padx=5)

        self.btn_stop = tk.Button(self.controls_frame, text="⏹ Stop", font=("Segoe UI", 10, "bold"), fg=self.COLOR_TEXT, bg=self.COLOR_BORDER,
                                  activebackground="#475569", activeforeground=self.COLOR_TEXT, bd=0, padx=12, pady=6, state="disabled", command=self.stop_recording)
        self.btn_stop.pack(side="left", padx=5)

        # Recordings Header
        self.header_label = tk.Label(self.container, text="Your Recordings", font=("Segoe UI", 11, "bold"), fg=self.COLOR_SUBTEXT, bg=self.COLOR_BG)

        # Recordings List Container
        self.list_frame = tk.Frame(self.container, bg=self.COLOR_BG)
        self.list_frame.pack(fill="both", expand=True)

    # --- Recording Operations ---

    def start_recording(self):
        self.is_recording = True
        self.is_paused = False
        self.audio_data = []
        self.elapsed_seconds = 0

        self.btn_record.config(state="disabled")
        self.btn_pause.config(state="normal", text="⏸ Pause")
        self.btn_stop.config(state="normal")
        self.status_label.config(text="Recording...")

        # Threads for Audio Capture & Timer Loop
        threading.Thread(target=self._record_audio_stream, daemon=True).start()
        threading.Thread(target=self._update_timer_loop, daemon=True).start()

    def _record_audio_stream(self):
        def callback(indata, frames, time_info, status):
            if self.is_recording and not self.is_paused:
                self.audio_data.append(indata.copy())
                # Calculate frequency intensity for live visualizer
                fft_data = np.abs(np.fft.rfft(indata[:, 0]))[:32]
                normalized = np.clip(fft_data / (np.max(fft_data) + 1e-5), 0, 1)
                self.root.after(0, self._draw_visualizer, normalized)

        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback):
            while self.is_recording:
                sd.sleep(100)

    def toggle_pause(self):
        if not self.is_paused:
            self.is_paused = True
            self.btn_pause.config(text="▶ Resume")
            self.status_label.config(text="Paused")
        else:
            self.is_paused = False
            self.btn_pause.config(text="⏸ Pause")
            self.status_label.config(text="Recording...")

    def stop_recording(self):
        self.is_recording = False
        self.is_paused = False

        self.btn_record.config(state="normal")
        self.btn_pause.config(state="disabled", text="⏸ Pause")
        self.btn_stop.config(state="disabled")
        self.status_label.config(text="Recording saved below")

        self._clear_visualizer()

        if self.audio_data:
            # Concatenate recorded audio arrays into float32 format
            recorded_array = np.concatenate(self.audio_data, axis=0).astype(np.float32)
            
            # Normalize audio signal
            max_val = np.max(np.abs(recorded_array))
            if max_val > 0:
                recorded_array = recorded_array / max_val

            # Convert normalized floats (-1.0 to 1.0) into 16-bit PCM integer range
            pcm16_data = (recorded_array * 32767).astype(np.int16)

            self.recording_count += 1
            filename = f"recording_{self.recording_count}.wav"
            wav.write(filename, self.sample_rate, pcm16_data)

            self._add_recording_card(filename, self.recording_count, self.elapsed_seconds)

    # --- UI Helpers & Visualizer ---

    def _draw_visualizer(self, values):
        for bar, val in zip(self.bars, values):
            bar.set_height(val)
        self.canvas_widget.draw_idle()

    def _clear_visualizer(self):
        for bar in self.bars:
            bar.set_height(0)
        self.canvas_widget.draw_idle()

    def _update_timer_loop(self):
        while self.is_recording:
            if not self.is_paused:
                time.sleep(1)
                self.elapsed_seconds += 1
                mins, secs = divmod(self.elapsed_seconds, 60)
                time_str = f"{mins:02d}:{secs:02d}"
                self.root.after(0, lambda t=time_str: self.timer_label.config(text=t))

    def _add_recording_card(self, filename, rec_id, duration):
        if not self.header_label.winfo_ismapped():
            self.header_label.pack(fill="x", pady=(5, 5))

        item_card = tk.Frame(self.list_frame, bg=self.COLOR_CARD, highlightbackground=self.COLOR_BORDER, highlightthickness=1)
        item_card.pack(fill="x", pady=5)

        info_frame = tk.Frame(item_card, bg=self.COLOR_CARD)
        info_frame.pack(fill="x", padx=10, pady=(8, 4))

        mins, secs = divmod(duration, 60)
        label_text = f"Recording {rec_id} ({mins:02d}:{secs:02d})"
        title_lbl = tk.Label(info_frame, text=label_text, font=("Segoe UI", 9, "bold"), fg=self.COLOR_TEXT, bg=self.COLOR_CARD)
        title_lbl.pack(side="left")

        # Action Buttons (Play / Delete)
        actions_frame = tk.Frame(info_frame, bg=self.COLOR_CARD)
        actions_frame.pack(side="right")

        btn_play = tk.Button(actions_frame, text="▶ Play", font=("Segoe UI", 8), fg=self.COLOR_TEXT, bg=self.COLOR_BORDER,
                             bd=0, padx=6, pady=2, command=lambda f=filename: self._play_audio(f))
        btn_play.pack(side="left", padx=2)

        btn_del = tk.Button(actions_frame, text="🗑 Delete", font=("Segoe UI", 8), fg="#f87171", bg=self.COLOR_CARD,
                            activeforeground="#dc2626", bd=1, relief="solid", padx=6, pady=2,
                            command=lambda c=item_card, f=filename: self._delete_recording(c, f))
        btn_del.pack(side="left", padx=2)

    def _play_audio(self, filename):
        def play_thread():
            if os.path.exists(filename):
                rate, data = wav.read(filename)
                
                # Convert 16-bit PCM back to float range (-1.0 to 1.0)
                if data.dtype == np.int16:
                    audio_float = data.astype(np.float32) / 32768.0
                else:
                    audio_float = data

                sd.play(audio_float, rate)
                sd.wait()  # Block background thread until playback finishes

        threading.Thread(target=play_thread, daemon=True).start()

    def _delete_recording(self, card_widget, filename):
        card_widget.destroy()
        if os.path.exists(filename):
            os.remove(filename)

        if len(self.list_frame.winfo_children()) == 0:
            self.header_label.pack_forget()
            self.status_label.config(text="Ready to record")


if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceRecorderApp(root)
    root.mainloop()