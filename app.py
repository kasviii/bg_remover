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
st.caption("Upload an image, draw a rectangle around your subject, and download the result with a transparent background.")

# ── sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    iterations = st.slider("GrabCut Iterations", 1, 15, 5,
                           help="More iterations = more accurate, slower")
    st.markdown("---")
    st.markdown("**How to use**")
    st.markdown("1. Upload a photo\n2. Set the crop rectangle using the sliders\n3. Click **Remove Background**\n4. Refine if needed\n5. Download the PNG")
    st.markdown("---")
    st.markdown("**How GrabCut works**")
    st.markdown(
        "- Your rectangle seeds the foreground region\n"
        "- Gaussian Mixture Models learn FG and BG colour distributions\n"
        "- Graph cut finds the optimal boundary\n"
        "- Iterates until convergence"
    )

# ── file upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp","webp"])

if uploaded is None:
    st.info("Upload an image to get started.")
    st.stop()

file_bytes = np.frombuffer(uploaded.read(), np.uint8)
img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# resize if too large
h, w = img.shape[:2]
if max(h, w) > 1200:
    s = 1200 / max(h, w)
    img = cv2.resize(img, (int(w*s), int(h*s)))
    h, w = img.shape[:2]

st.markdown("### Define the subject area")
st.caption("Use the sliders to draw a rectangle around your subject. Include a little margin around it.")

col1, col2 = st.columns(2)
with col1:
    x1 = st.slider("Left",   0, w-2, w//6)
    x2 = st.slider("Right",  x1+1, w, w - w//6)
with col2:
    y1 = st.slider("Top",    0, h-2, h//6)
    y2 = st.slider("Bottom", y1+1, h, h - h//6)

# show preview with rectangle
preview = img.copy()
cv2.rectangle(preview, (x1,y1), (x2,y2), (124,111,247), 2)
rgb_preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
st.image(rgb_preview, caption="Rectangle preview — adjust sliders until the box wraps your subject", use_container_width=True)

# ── run GrabCut ───────────────────────────────────────────────────────────────
if st.button("Remove Background", type="primary"):
    rect_w = x2 - x1
    rect_h = y2 - y1

    if rect_w < 10 or rect_h < 10:
        st.error("Rectangle too small — adjust the sliders.")
        st.stop()

    with st.spinner("Running GrabCut segmentation…"):
        mask       = np.zeros((h, w), dtype=np.uint8)
        bgd_model  = np.zeros((1, 65), dtype=np.float64)
        fgd_model  = np.zeros((1, 65), dtype=np.float64)
        rect       = (x1, y1, rect_w, rect_h)

        try:
            cv2.grabCut(img, mask, rect, bgd_model, fgd_model,
                        iterations, cv2.GC_INIT_WITH_RECT)
        except Exception as e:
            st.error(f"GrabCut failed: {e}")
            st.stop()

        fg_mask = np.where((mask == GC_FGD) | (mask == GC_PR_FGD), 255, 0).astype(np.uint8)

        # mask visualisation
        mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
        mask_vis[mask == GC_FGD]    = [80,  220, 80]
        mask_vis[mask == GC_PR_FGD] = [180, 220, 180]
        mask_vis[mask == GC_PR_BGD] = [80,  80,  100]
        mask_vis[mask == GC_BGD]    = [20,  20,  30]

        # checkerboard for transparency preview
        board = np.zeros((h, w, 3), dtype=np.uint8)
        sz = 16
        for y in range(0, h, sz):
            for x in range(0, w, sz):
                val = 200 if (x//sz + y//sz) % 2 == 0 else 230
                board[y:y+sz, x:x+sz] = val

        fg_3ch  = cv2.merge([fg_mask, fg_mask, fg_mask])
        cutout  = np.where(fg_3ch == 255,
                           cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                           board)

        # BGRA for export
        bgra        = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        bgra[:,:,3] = fg_mask

    # ── results ───────────────────────────────────────────────────────────────
    st.markdown("### Results")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                 caption="Original", use_container_width=True)

    with c2:
        st.image(mask_vis,
                 caption="Segmentation mask — green=foreground, dark=background",
                 use_container_width=True)

    with c3:
        st.image(cutout,
                 caption="Cutout (checkerboard = transparent)",
                 use_container_width=True)

    fg_pct = int(np.sum(fg_mask > 0) / fg_mask.size * 100)
    st.info(f"Foreground: {fg_pct}% of image  |  {np.sum(fg_mask > 0):,} pixels kept")

    # ── download ──────────────────────────────────────────────────────────────
    pil_img = Image.fromarray(cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    st.download_button(
        label="Download transparent PNG",
        data=buf,
        file_name="background_removed.png",
        mime="image/png",
        type="primary",
    )

    st.caption("Tip: if edges look rough, increase GrabCut Iterations in the sidebar and run again.")
