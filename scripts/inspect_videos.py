import cv2
from pathlib import Path

p = Path('in')
exts = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
files = [f for f in p.iterdir() if f.suffix.lower() in exts]
print('Found files:', [f.name for f in files])
for f in files:
    cap = cv2.VideoCapture(str(f))
    if not cap.isOpened():
        print(f.name, 'cannot be opened')
        continue
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f.name, 'frames=', frames, 'fps=', fps)
    cap.release()
