import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

GC_BGD    = cv2.GC_BGD
GC_FGD    = cv2.GC_FGD
GC_PR_BGD = cv2.GC_PR_BGD
GC_PR_FGD = cv2.GC_PR_FGD

st.set_page_config(page_title="Background Remover", layout="wide")
st.title("Background Remover")
st.caption("Upload an image, position the rectangle around your subject, then remove the background.")

with st.sidebar:
    st.header("Settings")
    iterations = st.slider("GrabCut Iterations", 1, 15, 5)
    st.markdown("---")
    st.markdown("**How to use**")
    st.markdown("1. Upload a photo\n2. Use the sliders to position the rectangle tightly around your subject\n3. Click **Remove Background**\n4. Download the transparent PNG")
    st.markdown("---")
    st.markdown("**How GrabCut works**")
    st.markdown(
        "- Rectangle seeds the foreground region\n"
        "- Gaussian Mixture Models learn FG and BG colour distributions\n"
        "- Graph cut finds the optimal boundary\n"
        "- Iterates until convergence"
    )

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp","webp"])

if uploaded is None:
    st.info("Upload an image to get started.")
    st.stop()

file_bytes = np.frombuffer(uploaded.read(), np.uint8)
img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

h, w = img_bgr.shape[:2]
if max(h, w) > 1200:
    s = 1200 / max(h, w)
    img_bgr = cv2.resize(img_bgr, (int(w*s), int(h*s)))
    h, w = img_bgr.shape[:2]

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ── rectangle sliders ─────────────────────────────────────────────────────────
st.markdown("### Position the rectangle around your subject")
st.caption("The purple rectangle updates live as you move the sliders. Get it as tight as possible around your subject.")

col1, col2 = st.columns(2)
with col1:
    left   = st.slider("Left edge",   0, w-2,  w//6,  key="left")
    right  = st.slider("Right edge",  left+1, w, w - w//6, key="right")
with col2:
    top    = st.slider("Top edge",    0, h-2,  h//6,  key="top")
    bottom = st.slider("Bottom edge", top+1, h, h - h//6, key="bottom")

# live preview with rectangle drawn on image
preview = img_rgb.copy()
cv2.rectangle(preview, (left, top), (right, bottom), (124, 111, 247), 3)
# dim outside the rectangle so subject stands out
overlay = preview.copy()
overlay[:top,  :] = (overlay[:top,  :] * 0.4).astype(np.uint8)
overlay[bottom:,:] = (overlay[bottom:,:] * 0.4).astype(np.uint8)
overlay[top:bottom, :left]  = (overlay[top:bottom, :left]  * 0.4).astype(np.uint8)
overlay[top:bottom, right:] = (overlay[top:bottom, right:] * 0.4).astype(np.uint8)

st.image(overlay, caption="Live preview — adjust sliders until the rectangle tightly wraps your subject", use_container_width=True)

rect_w = right - left
rect_h = bottom - top
st.caption(f"Rectangle: {rect_w}×{rect_h}px  |  Subject area: {rect_w*rect_h//(w*h//100)}% of image")

# ── run ───────────────────────────────────────────────────────────────────────
if not st.button("Remove Background", type="primary"):
    st.stop()

if rect_w < 20 or rect_h < 20:
    st.error("Rectangle too small — adjust the sliders.")
    st.stop()

with st.spinner("Running GrabCut segmentation…"):
    mask      = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(img_bgr, mask, (left, top, rect_w, rect_h),
                    bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    except Exception as e:
        st.error(f"GrabCut failed: {e}")
        st.stop()

    fg_mask = np.where(
        (mask == GC_FGD) | (mask == GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
    mask_vis[mask == GC_FGD]    = [80,  220, 80]
    mask_vis[mask == GC_PR_FGD] = [180, 220, 180]
    mask_vis[mask == GC_PR_BGD] = [80,  80,  100]
    mask_vis[mask == GC_BGD]    = [20,  20,  30]

    board = np.zeros((h, w, 3), dtype=np.uint8)
    sz = 16
    for y in range(0, h, sz):
        for x in range(0, w, sz):
            board[y:y+sz, x:x+sz] = 200 if (x//sz + y//sz) % 2 == 0 else 230

    fg_3ch = cv2.merge([fg_mask, fg_mask, fg_mask])
    cutout = np.where(fg_3ch == 255, img_rgb, board)

    bgra        = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:,:,3] = fg_mask

# ── results ───────────────────────────────────────────────────────────────────
st.markdown("### Results")
c1, c2, c3 = st.columns(3)
with c1:
    st.image(img_rgb,    caption="Original",           use_container_width=True)
with c2:
    st.image(mask_vis,   caption="Segmentation mask",  use_container_width=True)
with c3:
    st.image(cutout,     caption="Cutout (checkerboard = transparent)", use_container_width=True)

fg_pct = int(np.sum(fg_mask > 0) / fg_mask.size * 100)
st.info(f"Foreground: {fg_pct}% of image  |  {np.sum(fg_mask > 0):,} pixels kept")

pil_out = Image.fromarray(cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA))
buf = io.BytesIO()
pil_out.save(buf, format="PNG")
buf.seek(0)

st.download_button(
    label="Download transparent PNG",
    data=buf,
    file_name="background_removed.png",
    mime="image/png",
    type="primary",
)

st.caption("Tip: if edges look rough, increase GrabCut Iterations in the sidebar and run again.")
