"""DALM-DETR interactive demo (Gradio).

Side-by-side comparison of two CrowdHuman-trained detectors on any image:

    LEFT   : baseline        (standard Deformable-DETR Hungarian matching)
    RIGHT  : DALM-DETR       (density-adaptive location-aware matching)
    THIRD  : difference view (people found ONLY by DALM-DETR, in amber)

Runs on CPU (HF Spaces free tier). Checkpoints are loaded from local paths
if present (repo development), otherwise downloaded from the HF Hub model
repo given in the environment variables below.

Environment variables (used on HF Spaces):
    DALM_HF_REPO          e.g. "your-username/dalm-detr"   (model repo id)
    DALM_BASELINE_FILE    default "offsetiou_baseline_v3_best.pth"
    DALM_ADAPTIVE_FILE    default "offsetiou_ours_adaptive_v3_best.pth"
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Make the package importable both from repo root (demo/app.py) and from a
# HF Space where src/offsetiou_det is copied next to app.py.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
for cand in (_HERE / "src", _HERE.parent / "src", _HERE):
    if (cand / "offsetiou_det").is_dir():
        sys.path.insert(0, str(cand))
        break

# ---------------------------------------------------------------------------
# Model construction — mirrors scripts/train.py exactly:
#   OffsetIoUDet(num_classes=1, num_queries=300, pretrained_backbone=True)
# We pass pretrained_backbone=False here because the checkpoint fully
# overwrites all weights anyway — no need to download ImageNet weights on
# the Space. load_state_dict(strict=True) guarantees architecture match.
# ---------------------------------------------------------------------------
NUM_QUERIES = 300


def build_model() -> torch.nn.Module:
    from offsetiou_det.models.detector import OffsetIoUDet
    return OffsetIoUDet(
        num_classes=1,
        num_queries=NUM_QUERIES,
        pretrained_backbone=False,
    )


# ---------------------------------------------------------------------------
# Checkpoint resolution: local first, then HF Hub.
# ---------------------------------------------------------------------------
LOCAL_CKPT_DIR = _HERE.parent / "outputs" / "checkpoints"
HF_REPO = os.environ.get("DALM_HF_REPO", "")
CKPT_FILES = {
    "baseline": os.environ.get("DALM_BASELINE_FILE", "offsetiou_baseline_v3_best.pth"),
    "const": os.environ.get("DALM_CONST_FILE", "offsetiou_ours_const_v3_best.pth"),
    "dalm": os.environ.get("DALM_ADAPTIVE_FILE", "offsetiou_ours_adaptive_v3_best.pth"),
}

# dropdown display name -> checkpoint key (three-rung ablation, live)
MODEL_CHOICES = {
    "baseline · standard matching": "baseline",
    "ours-const · constant location prior": "const",
    "DALM-DETR · density-adaptive": "dalm",
}


def resolve_checkpoint(key: str) -> str:
    fname = CKPT_FILES[key]
    local = LOCAL_CKPT_DIR / fname
    if local.is_file():
        return str(local)
    if HF_REPO:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=HF_REPO, filename=fname)
    raise FileNotFoundError(
        f"Checkpoint '{fname}' not found locally ({local}) and DALM_HF_REPO is not set."
    )


_MODELS: dict[str, torch.nn.Module] = {}


def get_model(key: str) -> torch.nn.Module:
    if key not in _MODELS:
        model = build_model()
        try:
            state = torch.load(resolve_checkpoint(key), map_location="cpu", weights_only=True)
        except Exception:
            state = torch.load(resolve_checkpoint(key), map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
        model.load_state_dict(state, strict=True)
        model.eval()
        _MODELS[key] = model
    return _MODELS[key]


# ---------------------------------------------------------------------------
# Pre/post-processing (matches training: 640x640, ImageNet normalization,
# outputs pred_logits [B,Q,C] + pred_boxes [B,Q,4] normalized cx,cy,w,h).
# ---------------------------------------------------------------------------
IMAGE_SIZE = 640
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess(img: Image.Image) -> torch.Tensor:
    x = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(x)).permute(2, 0, 1).float() / 255.0
    return ((t - _MEAN) / _STD).unsqueeze(0)


@torch.no_grad()
def detect(key: str, img: Image.Image, conf: float):
    model = get_model(key)
    t0 = time.perf_counter()
    out = model(preprocess(img))
    dt = time.perf_counter() - t0

    # forward returns: (pred_logits, pred_boxes, aux_outputs)
    logits, boxes = out[0][0], out[1][0]
    if logits.shape[-1] >= 2:               # softmax over [person, ..., background]
        scores = logits.softmax(-1)[:, 0]
    else:                                    # single-logit sigmoid head
        scores = logits.sigmoid()[:, 0]

    keep = scores > conf
    scores, boxes = scores[keep], boxes[keep]

    W, H = img.size
    cx, cy, w, h = boxes.unbind(-1)
    xyxy = torch.stack(
        [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], dim=-1
    )
    return xyxy.numpy(), scores.numpy(), dt


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / np.clip(area_a + area_b - inter, 1e-6, None)


# ---------------------------------------------------------------------------
# Drawing — annotation-HUD style: thick corner brackets + mono score tags.
# ---------------------------------------------------------------------------
GREEN, AMBER, INK = (61, 220, 132), (255, 180, 84), (14, 20, 32)


def _font(size: int):
    for name in ("DejaVuSansMono.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_boxes(img: Image.Image, boxes, scores, color, dashed=False) -> Image.Image:
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    lw = max(2, round(min(out.size) / 320))
    fnt = _font(max(11, round(min(out.size) / 55)))
    for (x1, y1, x2, y2), s in zip(boxes, scores):
        if dashed:
            step = 3 * lw
            for xa in np.arange(x1, x2, 2 * step):
                d.line([(xa, y1), (min(xa + step, x2), y1)], fill=color, width=lw)
                d.line([(xa, y2), (min(xa + step, x2), y2)], fill=color, width=lw)
            for ya in np.arange(y1, y2, 2 * step):
                d.line([(x1, ya), (x1, min(ya + step, y2))], fill=color, width=lw)
                d.line([(x2, ya), (x2, min(ya + step, y2))], fill=color, width=lw)
        else:
            d.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        # corner brackets (the HUD signature)
        k = min(12 * lw, (x2 - x1) / 3, (y2 - y1) / 3)
        for cx_, cy_, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            d.line([(cx_, cy_), (cx_ + dx * k, cy_)], fill=color, width=lw * 2)
            d.line([(cx_, cy_), (cx_, cy_ + dy * k)], fill=color, width=lw * 2)
        tag = f"person {s:.2f}"
        tw, th = d.textbbox((0, 0), tag, font=fnt)[2:]
        ty = y1 - th - 4 if y1 - th - 4 > 0 else y1 + 2
        d.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=INK)
        d.text((x1 + 3, ty + 2), tag, fill=color, font=fnt)
    return out


def strata(n: int) -> str:
    return "sparse" if n < 10 else ("medium" if n < 25 else "dense")


# ---------------------------------------------------------------------------
# Main comparison pipeline.
# ---------------------------------------------------------------------------

def compare(img: Image.Image, conf: float, left_name: str, right_name: str):
    if img is None:
        raise gr.Error("Upload an image first.")

    try:
        bb, bs, bt = detect(MODEL_CHOICES[left_name], img, conf)
        ob, os_, ot = detect(MODEL_CHOICES[right_name], img, conf)
    except FileNotFoundError as e:
        raise gr.Error(f"Checkpoint not ready yet (training in progress): {e}")

    # people found only by DALM-DETR (no baseline box with IoU > 0.5)
    extra = np.arange(len(ob))
    if len(bb) and len(ob):
        extra = np.where(iou_matrix(ob, bb).max(axis=1) < 0.5)[0]

    left = draw_boxes(img, bb, bs, GREEN)
    right = draw_boxes(img, ob, os_, GREEN)
    diff = draw_boxes(img, ob[extra], os_[extra], AMBER, dashed=True)

    nb, no, ne = len(bb), len(ob), len(extra)
    stats = (
        f"| | {left_name.split(' \u00b7 ')[0]} | **{right_name.split(' \u00b7 ')[0]}** |\n|---|---|---|\n"
        f"| persons found | {nb} | **{no}** |\n"
        f"| scene density | {strata(nb)} | {strata(no)} |\n"
        f"| inference (CPU) | {bt:.2f}s | {ot:.2f}s |\n\n"
        + (f"**+{ne} additional person(s)** found only by the right model "
           f"(amber, third panel)." if ne else
           "Both models agree on this image — differences show up most in dense crowds.")
    )
    return left, right, diff, stats


# ---------------------------------------------------------------------------
# UI — dark surveillance-HUD theme, mono data labels, bracketed title.
# ---------------------------------------------------------------------------
CSS = """
:root { --ink:#0e1420; --panel:#161d2b; --line:#26304a;
        --green:#3ddc84; --amber:#ffb454; --txt:#e8ecf4; --mut:#8b94a7; }
.gradio-container { background:var(--ink)!important; color:var(--txt); }
#hud-title { text-align:center; padding:26px 0 4px; }
#hud-title .frame { display:inline-block; position:relative; padding:14px 34px; }
#hud-title .frame::before, #hud-title .frame::after,
#hud-title .frame i::before, #hud-title .frame i::after {
  content:""; position:absolute; width:18px; height:18px; border:3px solid var(--green); }
#hud-title .frame::before { top:0; left:0; border-right:0; border-bottom:0; }
#hud-title .frame::after  { top:0; right:0; border-left:0; border-bottom:0; }
#hud-title .frame i::before { bottom:0; left:0; border-right:0; border-top:0; }
#hud-title .frame i::after  { bottom:0; right:0; border-left:0; border-top:0; }
#hud-title h1 { margin:0; font-size:2rem; letter-spacing:.04em; color:var(--txt); }
#hud-title .tag { font-family:ui-monospace,monospace; color:var(--green);
  font-size:.8rem; letter-spacing:.08em; }
#hud-sub { text-align:center; color:var(--mut); max-width:760px; margin:6px auto 0; }
.panel-label { font-family:ui-monospace,monospace; letter-spacing:.06em;
  color:var(--mut); text-transform:uppercase; font-size:.75rem; }
#legend { font-family:ui-monospace,monospace; font-size:.8rem; color:var(--mut);
  text-align:center; margin-top:2px; }
#legend b.g { color:var(--green); } #legend b.a { color:var(--amber); }
footer { display:none!important; }
"""

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.green,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#0e1420",
    block_background_fill="#161d2b",
    block_border_color="#26304a",
    body_text_color="#e8ecf4",
)

HEADER = """
<div id="hud-title"><span class="frame"><i></i>
<h1>DALM-DETR</h1>
<span class="tag">person&nbsp;0.97&nbsp;·&nbsp;density-adaptive&nbsp;matching</span>
</span></div>
<p id="hud-sub">Density-Adaptive Location-Aware Matching for crowded pedestrian
detection. Same architecture, same loss — only the Hungarian assignment differs.
Upload a crowded scene and compare the standard matcher against ours.</p>
"""

LEGEND = ('<p id="legend"><b class="g">■ green</b> detections &nbsp;·&nbsp; '
          '<b class="a">▨ amber (dashed)</b> found only by DALM-DETR</p>')

EXAMPLES_DIR = _HERE / "examples"
EXAMPLES = sorted(str(p) for p in EXAMPLES_DIR.glob("*.jpg")) if EXAMPLES_DIR.is_dir() else []

with gr.Blocks(title="DALM-DETR — crowded pedestrian detection") as demo:
    gr.HTML(HEADER)
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="pil", label="input scene")
            conf = gr.Slider(0.05, 0.9, value=0.4, step=0.05, label="confidence threshold")
            choices = list(MODEL_CHOICES)
            left_sel = gr.Dropdown(choices, value=choices[0], label="left model")
            right_sel = gr.Dropdown(choices, value=choices[2], label="right model")
            btn = gr.Button("Detect people", variant="primary")
            if EXAMPLES:
                gr.Examples(EXAMPLES, inputs=inp, label="crowded examples")
            stats = gr.Markdown()
        with gr.Column(scale=2):
            with gr.Row():
                out_b = gr.Image(label="left model", elem_classes="panel-label")
                out_o = gr.Image(label="right model", elem_classes="panel-label")
            out_d = gr.Image(label="difference · found only by the right model")
            gr.HTML(LEGEND)
    gr.Markdown(
        "Trained on CrowdHuman (15k images) on a single RTX 4060 · "
        "primary metric MR⁻² · "
        "[code & paper-style README on GitHub](https://github.com/Sarvarbek-Erniyazov/DALM-DETR)"
    )
    btn.click(compare, inputs=[inp, conf, left_sel, right_sel],
              outputs=[out_b, out_o, out_d, stats])
    inp.change(lambda: None, None, None)  # no auto-run; keep CPU budget for the button

if __name__ == "__main__":
    demo.launch(theme=theme, css=CSS)
