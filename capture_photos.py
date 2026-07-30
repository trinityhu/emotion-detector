import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from typing import Optional

import cv2

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    tk = None
    ttk = None

try:
    import keyboard  # type: ignore[import-not-found]
except ImportError:
    keyboard = None


DEFAULT_SAVE_FOLDER = "/home/nvidia/jetson-inference/python/training/classification/data/emotions_split/test/sad"


def parse_args():
    parser = argparse.ArgumentParser(description="Capture images while holding the space bar.")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_FOLDER, help="Directory to save captured images")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index to use")
    parser.add_argument("--fps", type=float, default=10.0, help="Images per second while holding space")
    parser.add_argument("--preview", dest="preview", action="store_true", help="Show a preview window")
    parser.add_argument("--no-preview", dest="preview", action="store_false", help="Disable the preview window")
    parser.set_defaults(preview=True)
    return parser.parse_args()


def ensure_save_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_next_counter(save_dir):
    existing = []
    for entry in os.listdir(save_dir):
        if entry.startswith("img_") and entry.endswith(".jpg"):
            try:
                existing.append(int(entry.split("_")[1].split(".")[0]))
            except ValueError:
                continue
    return max(existing) + 1 if existing else 0


def get_camera_candidates(preferred_index=None):
    if preferred_index is None:
        return [0, 1, 2, 3]
    return [preferred_index, 0, 1, 2, 3]


def try_virtual_display():
    if os.environ.get("DISPLAY"):
        return True

    xvfb = shutil.which("Xvfb")
    if not xvfb:
        return False

    try:
        display_num = 99
        subprocess.Popen(
            [xvfb, f":{display_num}", "-screen", "0", "1280x720x24"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1)
        os.environ["DISPLAY"] = f":{display_num}"
        return True
    except Exception:
        return False


def open_camera(preferred_index=None):
    for index in get_camera_candidates(preferred_index):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            print(f"Using camera index {index}")
            return cap, index
        cap.release()

    for index in get_camera_candidates(preferred_index):
        path = f"/dev/video{index}"
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if cap.isOpened():
            print(f"Using camera device {path}")
            return cap, path
        cap.release()

    for pipeline in [
        "v4l2src device=/dev/video0 ! videoconvert ! appsink",
        "v4l2src device=/dev/video1 ! videoconvert ! appsink",
    ]:
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"Using GStreamer camera pipeline: {pipeline}")
            return cap, pipeline
        cap.release()

    return None, None


def read_key_nonblocking():
    kb = keyboard
    if kb is not None:
        try:
            if kb.is_pressed("space"):
                return "space"
            if kb.is_pressed("q"):
                return "q"
        except Exception:
            pass

    if not sys.stdin.isatty():
        return None

    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return None


def show_fallback_status(message):
    if tk is None or ttk is None:
        return False
    if not os.environ.get("DISPLAY"):
        return False

    try:
        root = tk.Tk()
        root.title("Dataset Collector")
        root.geometry("420x180")
        root.resizable(False, False)

        label = ttk.Label(root, text=message, wraplength=380, justify="center")
        label.pack(expand=True, padx=20, pady=20)

        button = ttk.Button(root, text="Close", command=root.destroy)
        button.pack(pady=(0, 20))

        root.mainloop()
        return True
    except Exception:
        return False


def save_frame(save_dir, counter, frame):
    filename = os.path.join(save_dir, f"img_{counter:05d}.jpg")
    ok = cv2.imwrite(filename, frame)
    if ok:
        print(f"[CAPTURE] Saved photo: {filename}")
    else:
        print(f"[CAPTURE] Failed to save photo: {filename}")
    return counter + 1


def should_stop_capture(stop_event):
    return stop_event is not None and stop_event.is_set()


def run_tk_button_window(save_dir, counter_ref, frame_ref, stop_event):
    if tk is None or ttk is None:
        return

    if not os.environ.get("DISPLAY"):
        return

    capturing = {"active": False}

    def capture_once():
        if frame_ref[0] is None:
            return
        counter_ref[0] = save_frame(save_dir, counter_ref[0], frame_ref[0])

    def start_capture():
        capturing["active"] = True
        capture_once()

    def stop_capture():
        capturing["active"] = False

    def capture_loop():
        while not stop_event.is_set():
            if capturing["active"]:
                capture_once()
            time.sleep(0.1)

    def close():
        stop_event.set()
        capturing["active"] = False
        root.destroy()

    root = tk.Tk()
    root.title("Capture Photo")
    root.geometry("220x140")
    root.resizable(False, False)

    button = ttk.Button(root, text="Hold to Capture", command=None)
    button.pack(expand=True, pady=(20, 8))

    def on_press(event=None):
        start_capture()

    def on_release(event=None):
        stop_capture()

    button.bind("<ButtonPress-1>", on_press)
    button.bind("<ButtonRelease-1>", on_release)
    button.bind("<Leave>", on_release)

    ttk.Button(root, text="Close", command=close).pack()

    thread = threading.Thread(target=capture_loop, daemon=True)
    thread.start()
    root.mainloop()


def main():
    args = parse_args()
    save_dir = ensure_save_dir(args.save_dir)
    counter = get_next_counter(save_dir)

    cap, used_index = open_camera(args.camera_index)
    if cap is None:
        print("Could not open any webcam.")
        print("This usually means the camera is already in use by another app or no camera is attached.")
        print("Try one of these:")
        print("  - Close any other program using your webcam")
        print("  - Check that the camera is connected")
        print("  - Try a different camera index with --camera-index 1")
        return 1

    preview_enabled = False
    if args.preview:
        if not os.environ.get("DISPLAY"):
            if not try_virtual_display():
                print("No display detected; preview window will require a desktop session or an X server.")

        try:
            cv2.namedWindow("Computer Webcam", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Computer Webcam", 960, 540)
            cv2.moveWindow("Computer Webcam", 100, 100)
            cv2.startWindowThread()
            preview_enabled = True
        except cv2.error as exc:
            print(f"Preview window is not available in this environment: {exc}")
            preview_enabled = False

    print(f"Saving images to: {save_dir}")
    print("A separate Capture window will open so you can save photos with a button.")
    print("Press Q to quit.")
    if preview_enabled:
        print(f"Preview window opened using camera index {used_index}.")
    else:
        if not os.environ.get("DISPLAY"):
            print("No display detected; running in terminal-only mode.")
        else:
            print("Preview window disabled or unavailable. Use --preview to try again.")
            show_fallback_status("The camera window could not be created.\nTry running the script from a desktop session.")

    if preview_enabled and os.environ.get("DISPLAY"):
        stop_event = threading.Event()
        frame_holder = [None]
        counter_holder = [counter]
        thread = threading.Thread(target=run_tk_button_window, args=(save_dir, counter_holder, frame_holder, stop_event), daemon=True)
        thread.start()
    else:
        stop_event = None
        frame_holder = None
        counter_holder = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to read from camera.")
            break

        if preview_enabled:
            if frame_holder is not None:
                frame_holder[0] = frame.copy()
            cv2.putText(frame, "CLICK THE CAPTURE BUTTON", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Computer Webcam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[QUIT] Exiting camera capture.")
                break
            if should_stop_capture(stop_event):
                break
        else:
            key = read_key_nonblocking()
            if key is not None:
                if key == "space" or key == ord(" "):
                    counter = save_frame(save_dir, counter, frame)
                elif key == ord("q") or (isinstance(key, str) and key.lower() == "q"):
                    print("[QUIT] Exiting camera capture.")
                    break
                else:
                    print(f"[KEY] Pressed key: {key}")
            else:
                time.sleep(0.05)

    cap.release()
    if preview_enabled:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
