# cutting_points.py
# Single-file implementation of Omni-cut (parallel decode + streaming embed + adaptive burst + capped heads)
# Exposes: get_cutting_points(video_path: str) -> List[float]
#
# Notes:
# - This file is self-contained and does NOT import any of your previous modules.
# - Behavior matches your original defaults; only an internal optimization in _decode_worker
#   uses cap.grab() for non-target frames inside a burst to reduce CPU copies while preserving outputs.
#
# Defaults are identical to your argparse block:
#   min_len=4.0
#   grids=[0.25, 0.5, 1.0]
#   novelty_hz=1.0
#   novelty_max_points=2400
#   use_gpu=True (i.e., --cpu flag is False by default)
#   max_feat_frames=200
#   embed_batch=256
#   sensitive=False
#   kts_lam_q=1.3
#   grid_bins_factor=8
#   decode_workers=8
#   head_workers=8
#   seek_mode="frame"
#   pelt_max_bins=256
#   burst_factor=1.25
#   min_burst_ms=250
#   max_burst_ms=3500
#   try_hwaccel=False

from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TORCH_HOME", r"/home/hice1/yxu846/scratch/models")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", r"/home/hice1/yxu846/scratch/models/hub")
os.environ.setdefault("KERAS_HOME", r"/home/hice1/yxu846/scratch/models")

import json
import math
import time
import queue
import warnings
import threading
from typing import List, Optional, Tuple, Dict

import numpy as np
import cv2
import ruptures as rpt

# Torch + torchvision are optional at run-time, but expected in your env
import torch
import torchvision.transforms as T

# timm is optional; if missing we disable embeddings
_HAS_TIMM = False
try:
    import timm
    _HAS_TIMM = True
except Exception:
    pass

# Avoid OpenCV multi-threading conflicts with our own pools and GPU
try:
    cv2.setNumThreads(0)
except Exception:
    pass


# ---------------- utils ----------------
def movmean(x, k):
    k = max(1, int(k))
    if k == 1:
        return x
    w = np.ones(k, dtype=np.float32) / k
    return np.convolve(x, w, mode="same")


def robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return (x - med) / (1.4826 * mad)


def cos_dist(a, b, eps=1e-8):
    return 1.0 - float(np.dot(a, b) / ((np.linalg.norm(a) + eps) * (np.linalg.norm(b) + eps)))


# ---------------- embedders ----------------
class BaseEmbedder:
    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def encode_batch(self, batch_bgr: List[np.ndarray], device="cuda", batch_size=64) -> np.ndarray:
        outs = [self(img) for img in batch_bgr]
        return np.stack(outs)


class DINOv2Embedder(BaseEmbedder):
    def __init__(self, device=None, model_name="vit_base_patch14_dinov2"):
        if not _HAS_TIMM:
            raise RuntimeError("timm not installed")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.perf_counter()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)
        self.tf = T.Compose([
            T.ToPILImage(),
            T.Resize((518, 518)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        t1 = time.perf_counter()
        print(f"[omnicut] DINOv2 embedder loaded on {self.device} in {t1 - t0:.2f}s")

    @torch.inference_mode()
    def stream_consumer(
        self,
        in_q: "queue.Queue[Optional[Tuple[int, np.ndarray]]]",
        out_map: Dict[int, np.ndarray],
        batch_size: int = 256,
    ):
        buf_idx: List[int] = []
        buf_tensors: List[torch.Tensor] = []

        def flush():
            if not buf_tensors:
                return
            x = torch.stack(buf_tensors).to(self.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                out = self.model(x.to(memory_format=torch.channels_last)).detach().cpu().numpy().astype(np.float32)
            for k, vec in zip(buf_idx, out):
                out_map[k] = vec
            buf_idx.clear()
            buf_tensors.clear()

        while True:
            item = in_q.get()
            if item is None:
                flush()
                in_q.task_done()
                break
            idx, bgr = item
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tens = self.tf(rgb)
            buf_idx.append(idx)
            buf_tensors.append(tens)
            in_q.task_done()
            if len(buf_tensors) >= batch_size:
                flush()
        return


# ---------------- low-level feats ----------------
def frame_hist_hsv(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    S = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    V = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    s = H.sum() + S.sum() + V.sum() + 1e-9
    return H / s, S / s, V / s


def frame_sharpness(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


class VideoFrames:
    def __init__(self, t, H, S, V, sharp, motion, emb, fps):
        self.t, self.H, self.S, self.V = t, H, S, V
        self.sharp, self.motion, self.emb, self.fps = sharp, motion, emb, fps


# ---------------- decode helpers ----------------
def _split_indices_even(idxs: np.ndarray, n_parts: int) -> List[np.ndarray]:
    if n_parts <= 1 or len(idxs) == 0:
        return [idxs]
    return list(np.array_split(idxs, n_parts))


def _decode_worker(
    video_path: str,
    idxs: np.ndarray,
    fps: float,
    resize_w: int,
    seek_mode: str,
    embed_q: "queue.Queue[Optional[Tuple[int, np.ndarray]]]",
    ff_threads: int = 1,
    try_hwaccel: bool = False,
    gap_frames: int = 8,
):
    """
    Burst reads: group nearby target frames (<= gap_frames apart); seek once per group,
    then advance with cheap cap.grab() and do cap.read() ONLY on target frames.
    This preserves target indices & outputs, reduces CPU copies, and reuses decoder cache.
    """
    gray_map, frame_map, hsv_map, sharp_map = {}, {}, {}, {}
    cap = cv2.VideoCapture(video_path, apiPreference=cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return gray_map, frame_map, hsv_map, sharp_map

    if hasattr(cv2, "CAP_PROP_THREADS"):
        try:
            cap.set(cv2.CAP_PROP_THREADS, float(ff_threads))
        except Exception:
            pass

    if try_hwaccel:
        try:
            if hasattr(cv2, "CAP_PROP_HW_ACCELERATION") and hasattr(cv2, "VIDEO_ACCELERATION_ANY"):
                cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            if hasattr(cv2, "CAP_PROP_HW_DEVICE"):
                cap.set(cv2.CAP_PROP_HW_DEVICE, 0)
        except Exception:
            pass

    # Build groups by frame gap (unchanged logic)
    groups = []
    cur = [int(idxs[0])]
    for a, b in zip(idxs[:-1], idxs[1:]):
        a = int(a)
        b = int(b)
        if (b - a) <= gap_frames:
            cur.append(b)
        else:
            groups.append(cur)
            cur = [b]
    if cur:
        groups.append(cur)

    def _seek_frame(fidx: int):
        if seek_mode == "msec":
            cap.set(cv2.CAP_PROP_POS_MSEC, float((fidx / max(fps, 1e-6)) * 1000.0))
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))

    for grp in groups:
        first = int(grp[0])
        _seek_frame(first)

        ok, f = cap.read()
        if not ok or f is None:
            continue

        # process first frame of group
        h, w = f.shape[:2]
        nh = int(h * resize_w / max(w, 1))
        fr = cv2.resize(f, (resize_w, nh), interpolation=cv2.INTER_LINEAR)
        h_, s_, v_ = frame_hist_hsv(fr)
        sharp = frame_sharpness(fr)
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

        hsv_map[first] = (h_.astype(np.float32), s_.astype(np.float32), v_.astype(np.float32))
        sharp_map[first] = float(sharp)
        gray_map[first] = g
        frame_map[first] = fr
        try:
            embed_q.put((first, fr), block=True)
        except Exception:
            pass

        prev = first

        # advance to subsequent targets by grabbing intermediate frames only
        for tgt in map(int, grp[1:]):
            skip = max(0, tgt - prev - 1)
            for _ in range(skip):
                if not cap.grab():
                    break
            ok, f = cap.read()
            if not ok or f is None:
                break

            h, w = f.shape[:2]
            nh = int(h * resize_w / max(w, 1))
            fr = cv2.resize(f, (resize_w, nh), interpolation=cv2.INTER_LINEAR)
            h_, s_, v_ = frame_hist_hsv(fr)
            sharp = frame_sharpness(fr)
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

            hsv_map[tgt] = (h_.astype(np.float32), s_.astype(np.float32), v_.astype(np.float32))
            sharp_map[tgt] = float(sharp)
            gray_map[tgt] = g
            frame_map[tgt] = fr
            try:
                embed_q.put((tgt, fr), block=True)
            except Exception:
                pass

            prev = tgt

    cap.release()
    return gray_map, frame_map, hsv_map, sharp_map


# ---------------- FAST sampled decode (parallel + streaming embed + adaptive burst) ----------------
def decode_sampled(
    video_path: str,
    max_frames: int = 200,
    resize_w: int = 384,
    embedder: Optional[DINOv2Embedder] = None,
    embed_batch: int = 256,
    decode_workers: int = 8,
    seek_mode: str = "frame",
    burst_factor: float = 1.25,  # scale median sample-gap to define group size
    min_burst_ms: int = 250,
    max_burst_ms: int = 3500,
    try_hwaccel: bool = False,
) -> VideoFrames:
    start = time.perf_counter()
    cap0 = cv2.VideoCapture(video_path, apiPreference=cv2.CAP_FFMPEG)
    if not cap0.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap0.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap0.release()

    # fully sequential fallback if frame count unknown
    if total <= 0:
        cap = cv2.VideoCapture(video_path, apiPreference=cv2.CAP_FFMPEG)
        frames, ts = [], []
        i = 0
        while len(frames) < max_frames:
            ok, f = cap.read()
            if not ok:
                break
            ts.append(i / fps)
            frames.append(f)
            i += 1
        cap.release()

        if not frames:
            return VideoFrames(np.array([]), *(np.array([]) for _ in range(6)), fps)

        H_list, S_list, V_list, sharp_list, gray_list = [], [], [], [], []
        embed_map: Dict[int, np.ndarray] = {}
        q = queue.Queue(maxsize=max(1, embed_batch * 2))

        t_cons = None
        if embedder is not None:
            t_cons = threading.Thread(
                target=embedder.stream_consumer, args=(q, embed_map, embed_batch), daemon=True
            )
            t_cons.start()

        for i, f in enumerate(frames):
            h, w = f.shape[:2]
            nh = int(h * resize_w / max(w, 1))
            fr = cv2.resize(f, (resize_w, nh))
            h_, s_, v_ = frame_hist_hsv(fr)
            H_list.append(h_)
            S_list.append(s_)
            V_list.append(v_)
            sharp_list.append(frame_sharpness(fr))
            gray_list.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
            if embedder is not None:
                q.put((i, fr))
        if embedder is not None:
            q.put(None)
            t_cons.join()

        H = np.stack(H_list).astype(np.float32)
        S = np.stack(S_list).astype(np.float32)
        V = np.stack(V_list).astype(np.float32)
        sharp = np.array(sharp_list, dtype=np.float32)
        motion = np.zeros(len(gray_list), dtype=np.float32)
        for j in range(1, len(gray_list)):
            diff = np.median(np.abs(gray_list[j].astype(np.int16) - gray_list[j - 1].astype(np.int16)))
            motion[j] = float(diff) / 255.0
        emb = None
        if embedder is not None and embed_map:
            emb = np.stack([embed_map[i] for i in range(len(frames))], 0).astype(np.float32)
        t = np.array(ts, dtype=np.float32)
        print(f"[omnicut] reading done in {time.perf_counter() - start:.2f}s for {len(frames)} frames")
        return VideoFrames(t, H, S, V, sharp, motion, emb, fps)

    # Choose uniformly spaced frame indices
    k = min(max_frames, total)
    if k <= 0:
        return VideoFrames(np.array([]), *(np.array([]) for _ in range(6)), fps)
    idxs = np.linspace(0, total - 1, num=k, dtype=int)

    # Adaptive burst size (in frames) from sample spacing
    if len(idxs) >= 2:
        median_gap_frames = int(max(1, np.median(np.diff(idxs))))
    else:
        median_gap_frames = 1
    desired_gap_ms = int((median_gap_frames / max(fps, 1e-6)) * 1000.0 * float(burst_factor))
    burst_ms = int(min(max(desired_gap_ms, int(min_burst_ms)), int(max_burst_ms)))
    gap_frames = max(1, int(round((burst_ms / 1000.0) * max(fps, 1e-6))))

    # Shared containers
    gray_map: Dict[int, np.ndarray] = {}
    frame_map: Dict[int, np.ndarray] = {}
    hsv_map: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    sharp_map: Dict[int, float] = {}
    embed_map: Dict[int, np.ndarray] = {}

    # Bounded queue to stream frames to GPU consumer
    embed_q: "queue.Queue[Optional[Tuple[int, np.ndarray]]]" = queue.Queue(maxsize=max(1, embed_batch * 2))

    # Start embedding consumer thread
    consumer = None
    if embedder is not None:
        consumer = threading.Thread(
            target=embedder.stream_consumer, args=(embed_q, embed_map, embed_batch), daemon=True
        )
        consumer.start()

    # Launch decode workers
    parts = _split_indices_even(idxs, max(1, decode_workers))
    from concurrent.futures import ThreadPoolExecutor, as_completed

    futures = []
    with ThreadPoolExecutor(max_workers=max(1, decode_workers)) as ex:
        for p in parts:
            if len(p) == 0:
                continue
            futures.append(
                ex.submit(
                    _decode_worker,
                    video_path,
                    p,
                    fps,
                    resize_w,
                    seek_mode,
                    embed_q,
                    1,
                    try_hwaccel,
                    gap_frames,
                )
            )
        for fu in as_completed(futures):
            gm, fm, hm, sm = fu.result()
            gray_map.update(gm)
            frame_map.update(fm)
            hsv_map.update(hm)
            sharp_map.update(sm)

    # All workers finished; stop consumer
    if embedder is not None:
        embed_q.put(None)
        consumer.join()

    end = time.perf_counter()
    print(f"[omnicut] reading done in {end - start:.2f}s for {len(idxs)} frames")

    # Assemble in order of idxs
    kept_idxs = [i for i in idxs.tolist() if i in hsv_map]
    if not kept_idxs:
        return VideoFrames(np.array([]), *(np.array([]) for _ in range(6)), fps)

    H = np.stack([hsv_map[i][0] for i in kept_idxs]).astype(np.float32)
    S = np.stack([hsv_map[i][1] for i in kept_idxs]).astype(np.float32)
    V = np.stack([hsv_map[i][2] for i in kept_idxs]).astype(np.float32)
    sharp = np.array([sharp_map[i] for i in kept_idxs], dtype=np.float32)
    t = np.array([float(i) / max(fps, 1e-6) for i in kept_idxs], dtype=np.float32)

    # Motion from consecutive gray frames
    gs = [gray_map[i] for i in kept_idxs]
    motion = np.zeros(len(gs), dtype=np.float32)
    for j in range(1, len(gs)):
        diff = np.median(np.abs(gs[j].astype(np.int16) - gs[j - 1].astype(np.int16)))
        motion[j] = float(diff) / 255.0

    emb = None
    if embedder is not None and embed_map:
        emb = np.stack([embed_map[i] for i in kept_idxs if i in embed_map], 0).astype(np.float32)
        if emb.shape[0] != len(kept_idxs):
            common = [i for i in kept_idxs if i in embed_map]
            H = np.stack([hsv_map[i][0] for i in common]).astype(np.float32)
            S = np.stack([hsv_map[i][1] for i in common]).astype(np.float32)
            V = np.stack([hsv_map[i][2] for i in common]).astype(np.float32)
            sharp = np.array([sharp_map[i] for i in common], dtype=np.float32)
            t = np.array([float(i) / max(fps, 1e-6) for i in common], dtype=np.float32)
            gs = [gray_map[i] for i in common]
            motion = np.zeros(len(gs), dtype=np.float32)
            for j in range(1, len(gs)):
                diff = np.median(np.abs(gs[j].astype(np.int16) - gs[j - 1].astype(np.int16)))
                motion[j] = float(diff) / 255.0

    return VideoFrames(t, H, S, V, sharp, motion, emb, fps)


# ---------------- fused per-step ----------------
def build_perstep_fused(vf: VideoFrames, ab: Optional[dict], ao: Optional[dict], ema_alpha=0.9):
    t = vf.t
    n = len(t)
    if n < 2:
        return t, np.zeros(0, dtype=np.float32)

    def cosrow(A, B):
        num = np.sum(A * B, axis=1)
        den = (np.linalg.norm(A, axis=1) + 1e-8) * (np.linalg.norm(B, axis=1) + 1e-8)
        return 1.0 - (num / den)

    dH = cosrow(vf.H[1:], vf.H[:-1])
    dS = cosrow(vf.S[1:], vf.S[:-1])
    dV = cosrow(vf.V[1:], vf.V[:-1]) * 0.1
    col = 0.55 * dH + 0.35 * dS + 0.10 * dV

    shp = np.abs(np.diff(vf.sharp))
    mot = np.abs(np.diff(vf.motion))

    if vf.emb is not None and vf.emb.shape[0] == n:
        e = vf.emb
        d_prev = np.array([cos_dist(e[i], e[i - 1]) for i in range(1, n)], dtype=np.float32)
        ema = np.zeros_like(e, dtype=np.float32)
        ema[0] = e[0]
        for i in range(1, n):
            ema[i] = ema_alpha * ema[i - 1] + (1 - ema_alpha) * e[i]
        d_ema = np.array([cos_dist(e[i], ema[i - 1]) for i in range(1, n)], dtype=np.float32)
        emb = 0.5 * d_prev + 0.5 * d_ema
    else:
        emb = np.zeros(n - 1, dtype=np.float32)

    t_mid = 0.5 * (t[1:] + t[:-1])
    aud = np.zeros_like(col, dtype=np.float32)

    def rz(v):
        return np.clip(robust_z(movmean(v, 5)), -4, 4).astype(np.float32)

    z_col, z_mot, z_emb, z_aud, z_shp = map(rz, [col, mot, emb, aud, shp])

    vars_ = np.array([z_col.std(), z_mot.std(), z_emb.std(), z_aud.std(), z_shp.std()], dtype=np.float32) + 1e-6
    base = np.array([0.20, 0.30, 0.35, 0.10, 0.05], dtype=np.float32)
    w = base * (vars_ / vars_.sum())
    w = w / w.sum()
    fused = w[0] * z_col + w[1] * z_mot + w[2] * z_emb + w[3] * z_aud + w[4] * z_shp
    fused = movmean(fused, 5).astype(np.float32)
    return t_mid.astype(np.float32), fused


# ---------------- pooling & SSM ----------------
def pool_to_fixed_grid(t, s, bin_sec, max_bins=None):
    if len(s) == 0:
        return np.array([]), np.array([])
    t0, t1 = float(t[0]), float(t[-1])
    span = max(t1 - t0, 1e-6)
    nb = int(np.ceil(span / bin_sec))
    if max_bins is not None and nb > max_bins:
        nb = max_bins
        edges = np.linspace(t0, t1, nb + 1, dtype=np.float32)
    else:
        edges = np.arange(t0, t1 + 1e-6, bin_sec, dtype=np.float32)
        nb = len(edges) - 1
    if nb < 2:
        return np.array([]), np.array([])
    b = np.digitize(t, edges) - 1
    pooled = np.zeros(nb, dtype=np.float32)
    for i in range(nb):
        m = (b == i)
        pooled[i] = np.max(s[m]) if np.any(m) else 0.0
    t_bin = 0.5 * (edges[:-1] + edges[1:])
    return t_bin, pooled


def embedding_to_grid(t, emb, grid_sec, fill=True):
    if emb is None or len(t) == 0:
        return np.array([]), np.empty((0, 0), dtype=np.float32)
    t0, t1 = float(t[0]), float(t[-1])
    edges = np.arange(t0, t1 + grid_sec + 1e-9, grid_sec, dtype=np.float32)
    if len(edges) < 2:
        return np.array([]), np.empty((0, emb.shape[1]), dtype=emb.dtype)
    b = np.digitize(t, edges) - 1
    nb = len(edges) - 1
    tg = 0.5 * (edges[:-1] + edges[1:])
    Eg = []
    last = None
    for i in range(nb):
        m = (b == i)
        if np.any(m):
            v = emb[m].mean(axis=0)
            last = v
            Eg.append(v)
        else:
            Eg.append(last if (fill and last is not None) else np.zeros((emb.shape[1],), emb.dtype))
    return tg, np.stack(Eg).astype(np.float32)


def novelty_from_embeddings(E, win=16, sigma=8.0):
    if E is None or len(E) == 0:
        return np.zeros(0, dtype=np.float32)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    S = En @ En.T
    n = S.shape[0]
    if n < 2 * win + 2:
        return np.zeros(n, dtype=np.float32)
    x = np.arange(-win, win, dtype=np.float32)
    g = np.exp(-(x**2) / (2.0 * sigma**2)).astype(np.float32)
    G = np.outer(g, g)
    K = np.block(
        [
            [np.ones((win, win), np.float32), -np.ones((win, win), np.float32)],
            [-np.ones((win, win), np.float32), np.ones((win, win), np.float32)],
        ]
    ) * G
    nov = np.zeros(n, dtype=np.float32)
    for i in range(win, n - win):
        A = S[i - win : i, i - win : i]
        B = S[i - win : i, i : i + win]
        C = S[i : i + win, i - win : i]
        D = S[i : i + win, i : i + win]
        M = np.block([[A, B], [C, D]])
        nov[i] = float(np.sum(M * K))
    mu, sd = nov.mean(), nov.std() + 1e-9
    return ((nov - mu) / sd).astype(np.float32)


def novelty_mean_split(E, win=16):
    n, d = E.shape
    if n < 2 * win + 2:
        return np.zeros(n, dtype=np.float32)
    cs = np.cumsum(E, axis=0, dtype=np.float64)
    cs = np.vstack([np.zeros((1, d), dtype=np.float64), cs])
    centers = np.arange(win, n - win, dtype=int)
    L = (cs[centers] - cs[centers - win]) / float(win)
    R = (cs[centers + win] - cs[centers]) / float(win)
    Ln = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-9)
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    cos = (Ln * Rn).sum(axis=1)
    dist = 1.0 - cos
    z = (dist - dist.mean()) / (dist.std() + 1e-9)
    nov = np.zeros(n, dtype=np.float32)
    nov[centers] = z.astype(np.float32)
    return nov


# ---------------- KTS ----------------
def _gram_embeddings(E: np.ndarray) -> np.ndarray:
    E = E.astype(np.float32, copy=False)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    return E @ E.T


def _integral_image(M: np.ndarray) -> np.ndarray:
    CS = np.zeros((M.shape[0] + 1, M.shape[1] + 1), dtype=np.float64)
    CS[1:, 1:] = np.cumsum(np.cumsum(M, axis=0), axis=1)
    return CS


def _rect_sum(CS: np.ndarray, i: int, j: int) -> float:
    return float(CS[j + 1, j + 1] - CS[i, j + 1] - CS[j + 1, i] + CS[i, i])


def _seg_cost(CS: np.ndarray, i: int, j: int) -> float:
    L = j - i + 1
    if L <= 0:
        return 0.0
    Sim = _rect_sum(CS, i, j)
    return -Sim / (L * L)


def kts_greedy(t: np.ndarray, E: np.ndarray, min_len_sec: float, lam_q: float = 1.4, max_splits: int = 30) -> List[float]:
    n = E.shape[0]
    if n < 3:
        return []
    S = _gram_embeddings(E)
    CS = _integral_image(S)

    def valid_k(i, j, k):
        return (t[k] - t[i] >= min_len_sec) and (t[j] - t[k] >= min_len_sec)

    base_cost = _seg_cost(CS, 0, n - 1)
    lam = max(1e-6, lam_q * abs(base_cost))
    cuts = []
    stack = [(0, n - 1)]
    while stack and len(cuts) < max_splits:
        i, j = stack.pop()
        if j - i + 1 < 3:
            continue
        best_gain = -1e9
        best_k = None
        cij = _seg_cost(CS, i, j)
        for k in range(i + 1, j):
            if not valid_k(i, j, k):
                continue
            gain = cij - (_seg_cost(CS, i, k) + _seg_cost(CS, k + 1, j)) - lam
            if gain > best_gain:
                best_gain = gain
                best_k = k
        if best_k is not None and best_gain > 0:
            cuts.append(best_k)
            stack.append((best_k + 1, j))
            stack.append((i, best_k))
    return [float(t[k]) for k in sorted(set(cuts))]


# ---------------- segmentation (PELT) ----------------
def boundary_strength(x, i, w):
    L = max(0, i - w)
    R = min(len(x), i + w)
    left = x[L:i].mean() if i > L else 0.0
    right = x[i:R].mean() if R > i else 0.0
    return abs(right - left)


def two_sample_best_split(x, min_gap):
    n = len(x)
    if n < 2 * min_gap + 1:
        return None
    cs = np.cumsum(x)
    best, bi = -1.0, None
    for i in range(min_gap, n - min_gap):
        mL = cs[i] / i
        mR = (cs[-1] - cs[i]) / (n - i)
        v = abs(mR - mL)
        if v > best:
            best, bi = v, i
    return bi


def segment_on_grid(tg, sg, min_len_sec, pen_scale=(1.0, 5.0), strength_q=0.7, model="auto"):
    if len(sg) < 5:
        return []
    z = np.clip(robust_z(sg), -4, 4)
    dt = float(np.median(np.diff(tg)))
    dt = max(dt, 1e-3)
    min_size = max(3, int(round(min_len_sec / dt)))
    total = tg[-1] - tg[0]
    if total < 2 * min_len_sec:
        return []

    n = len(z)
    mdl = "rbf" if n <= 1000 else "l2"
    sigma2 = float(np.mean(np.clip(z, -3, 3) ** 2))
    base_pen = sigma2 * np.log(max(3, n))
    lo, hi = base_pen * pen_scale[0], base_pen * pen_scale[1]
    pen_vals = np.geomspace(max(1e-3, lo), max(hi, lo * 1.01), 9)

    X = z.reshape(-1, 1).astype(np.float32)
    algo = rpt.Pelt(model=mdl, min_size=min_size).fit(X)
    cands = []
    for pen in pen_vals:
        bkps = np.array(algo.predict(pen=float(pen)), dtype=int)
        idx = [0] + bkps.tolist()
        lens = [(tg[idx[i + 1] - 1] - tg[idx[i]]) for i in range(len(idx) - 1)]
        minseg = min(lens) if lens else np.inf
        k = len(bkps) - 1
        cands.append(dict(bkps=bkps, k=k, minseg=minseg))
    feas = [c for c in cands if c["minseg"] >= min_len_sec]
    chosen = min(feas, key=lambda c: c["k"]) if feas else max(cands, key=lambda c: c["minseg"])
    bk = chosen["bkps"][:-1]

    if bk.size:
        W = max(4, min_size // 2)
        strengths = np.array([boundary_strength(z, i, W) for i in bk])
        thr = np.quantile(strengths, strength_q) if bk.size >= 4 else strengths.min()
        bk = bk[strengths >= thr]

    if bk.size == 0:
        q = max(1, len(z) // 4)
        drift = abs(z[-q:].mean() - z[:q].mean())
        idx = two_sample_best_split(z, min_size)
        if idx is not None and drift >= 0.5:
            bk = np.array([idx], dtype=int)

    return [float(tg[min(i, len(tg) - 1)]) for i in np.sort(bk)]


# ---------------- omni-cut (parallel heads) ----------------
def _run_pelt_for_grid(args):
    tg, sg, min_len_sec = args
    return segment_on_grid(tg, sg, min_len_sec=min_len_sec, model="auto")

device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = None
if _HAS_TIMM:
    try:
        embedder = DINOv2Embedder(device=device)
    except Exception as e:
        warnings.warn(f"Embedder init failed ({e}); disabling embeddings.")
        embedder = None

def omni_cut(
    video_path: str,
    min_len_sec=2.0,
    grid_levels=(0.25, 0.5, 1.0),
    target_len_sec: Optional[float] = None,
    novelty_hz=1.0,
    novelty_max_points=2400,
    use_gpu=True,
    max_feat_frames=200,
    embed_batch=256,
    sensitive=False,
    kts_lam_q=1.4,
    grid_bins_factor=8,
    decode_workers: int = 8,
    head_workers: int = 8,
    seek_mode: str = "frame",
    pelt_max_bins: int = 256,
    burst_factor: float = 1.25,
    min_burst_ms: int = 250,
    max_burst_ms: int = 3500,
    try_hwaccel: bool = False,
) -> List[float]:
    # torch perf knobs
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    global embedder

    t0 = time.perf_counter()
    vf = decode_sampled(
        video_path,
        max_frames=max_feat_frames,
        resize_w=384,
        embedder=embedder,
        embed_batch=embed_batch,
        decode_workers=decode_workers,
        seek_mode=seek_mode,
        burst_factor=burst_factor,
        min_burst_ms=min_burst_ms,
        max_burst_ms=max_burst_ms,
        try_hwaccel=try_hwaccel,
    )

    min_len_sec = max(vf.t[-1] / 15, min_len_sec)

    # target length sec is video length / 6
    target_len_sec = vf.t[-1] / 6 if target_len_sec is None and len(vf.t) >= 2 else target_len_sec

    if len(vf.t) < 2:
        t_end = time.perf_counter()
        print(f"  [omni-cut] decode+embed {t_end - t0:.2f}s | heads+post 0.00s | total {t_end - t0:.2f}s | cuts=0")
        return []

    t1 = time.perf_counter()
    t_mid, fused = build_perstep_fused(vf, ab=None, ao=None)

    # cap PELT per-grid size (stabilizes long-video latency)
    max_bins = min(pelt_max_bins, len(fused))

    pelt_jobs = []
    for g in grid_levels:
        tg, sg = pool_to_fixed_grid(t_mid, fused, bin_sec=g, max_bins=max_bins)
        if len(sg) == 0:
            continue
        pelt_jobs.append((tg, sg, min_len_sec))

    all_cuts: List[float] = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max(1, head_workers)) as ex:
        futures = []
        for job in pelt_jobs:
            futures.append(ex.submit(_run_pelt_for_grid, job))

        def _novelty_task():
            cuts_local = []
            if vf.emb is not None and vf.emb.shape[0] == len(vf.t) and novelty_hz > 0:
                total = float(vf.t[-1] - vf.t[0])
                grid_sec = max(1.0 / novelty_hz, 0.25)
                max_pts = int(min(novelty_max_points, max(128, 4 * len(vf.t))))
                if total / grid_sec > max_pts:
                    grid_sec = max(total / max_pts, grid_sec)
                tg_emb, Eg = embedding_to_grid(vf.t, vf.emb, grid_sec, fill=True)
                n_ssm = len(tg_emb)
                if n_ssm >= 12:
                    if n_ssm <= 600:
                        win = max(8, int(round((1.5 if sensitive else 2.0) / grid_sec)))
                        sigma = max(4.0, (1.2 if sensitive else 1.5) / grid_sec)
                        nov = novelty_from_embeddings(Eg, win=win, sigma=sigma)
                    else:
                        win = max(8, int(round((1.5 if sensitive else 2.0) / grid_sec)))
                        nov = novelty_mean_split(Eg, win=win)
                    z = np.clip(robust_z(nov), -4, 4).astype(np.float32)
                    dtg = float(np.median(np.diff(tg_emb)))
                    dtg = max(dtg, 1e-3)
                    min_gap_idx = max(3, int(round(min_len_sec / dtg)))
                    thr = np.quantile(z, 0.70 if sensitive else 0.80)
                    peaks = []
                    last = -10**9
                    for i in range(1, len(z) - 1):
                        if z[i] > z[i - 1] and z[i] > z[i + 1] and z[i] >= thr:
                            if i - last >= min_gap_idx:
                                peaks.append(i)
                                last = i
                    cuts_local.extend([float(tg_emb[i]) for i in peaks])
            return cuts_local

        def _kts_task():
            if vf.emb is not None and len(vf.emb) == len(vf.t):
                return kts_greedy(vf.t, vf.emb, min_len_sec=min_len_sec, lam_q=kts_lam_q)
            return []

        futures.append(ex.submit(_novelty_task))
        futures.append(ex.submit(_kts_task))

        for fu in as_completed(futures):
            all_cuts.extend(fu.result())

    # NMS with min-gap
    all_cuts = sorted(all_cuts)
    pruned = []
    last = -1e-9
    for c in all_cuts:
        if c - last >= min_len_sec:
            pruned.append(c)
            last = c
    all_cuts = pruned

    # cap by target length (optional)
    if target_len_sec is not None and len(all_cuts) > 0:
        total = vf.t[-1] - vf.t[0]
        max_cuts = max(0, int(math.floor(total / max(target_len_sec, 1e-3))) - 1)
        if len(all_cuts) > max_cuts:
            tg, sg = pool_to_fixed_grid(t_mid, fused, bin_sec=0.5, max_bins=max_bins)
            if len(sg) > 0:
                z = np.clip(robust_z(sg), -4, 4)
                dt = float(np.median(np.diff(tg)))
                dt = max(dt, 1e-3)
                W = max(4, int(round(min_len_sec / dt) // 2))
                idxs = [int(np.argmin(np.abs(tg - c))) for c in all_cuts]
                strengths = np.array([boundary_strength(z, i, W) for i in idxs])
                keep = strengths.argsort()[-max_cuts:]
                all_cuts = sorted([all_cuts[i] for i in keep])

    t2 = time.perf_counter()
    print(f"  [omni-cut] decode+embed {t1 - t0:.2f}s | heads+post {t2 - t1:.2f}s | total {t2 - t0:.2f}s | cuts={len(all_cuts)}")
    return all_cuts


# ---------------- public API ----------------
def get_cutting_points(video_path: str) -> List[float]:
    """
    Compute cut points for a video using the EXACT same defaults
    as the original argparse section. Only 'video_path' is required.
    """
    # Argparse defaults (do NOT change)
    min_len = 4.0
    target_len = None
    grids = (0.25, 0.5, 1.0)
    novelty_hz = 1.0
    novelty_max_points = 2400
    use_gpu = True
    max_feat_frames = 200
    embed_batch = 256
    sensitive = False
    kts_lam_q = 1.3  # important: matches argparse default even if omni_cut default differs
    grid_bins_factor = 8
    decode_workers = 8
    head_workers = 8
    seek_mode = "frame"
    pelt_max_bins = 256
    burst_factor = 1.25
    min_burst_ms = 250
    max_burst_ms = 3500
    try_hwaccel = False

    cuts = omni_cut(
        video_path,
        min_len_sec=min_len,
        grid_levels=grids,
        target_len_sec=target_len,
        novelty_hz=novelty_hz,
        novelty_max_points=novelty_max_points,
        use_gpu=use_gpu,
        max_feat_frames=max_feat_frames,
        embed_batch=embed_batch,
        sensitive=sensitive,
        kts_lam_q=kts_lam_q,
        grid_bins_factor=grid_bins_factor,
        decode_workers=decode_workers,
        head_workers=head_workers,
        seek_mode=seek_mode,
        pelt_max_bins=pelt_max_bins,
        burst_factor=burst_factor,
        min_burst_ms=min_burst_ms,
        max_burst_ms=max_burst_ms,
        try_hwaccel=try_hwaccel,
    )
    return cuts
