import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import io

GC_BGD    = cv2.GC_BGD
GC_FGD    = cv2.GC_FGD
GC_PR_BGD = cv2.GC_PR_BGD
GC_PR_FGD = cv2.GC_PR_FGD

st.set_page_config(page_title="Background Remover", layout="wide")
st.title("Background Remover")
st.caption("Draw a rectangle around your subject to remove the background using GrabCut segmentation.")

with st.sidebar:
    st.header("Settings")
    iterations = st.slider("GrabCut Iterations", 1, 15, 5,
                           help="More = more accurate, slower")
    st.markdown("---")
    st.markdown("**How to use**")
    st.markdown("1. Upload a photo\n2. Draw a rectangle around your subject\n3. Click **Remove Background**\n4. Download the transparent PNG")
    st.markdown("---")
    st.markdown("**How GrabCut works**")
    st.markdown(
        "- Rectangle seeds the foreground region\n"
        "- Gaussian Mixture Models learn FG and BG colour distributions\n"
        "- Graph cut finds the optimal boundary between them\n"
        "- Iterates until convergence"
    )

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp","webp"])

if uploaded is None:
    st.info("Upload an image to get started.")
    st.stop()

file_bytes = np.frombuffer(uploaded.read(), np.uint8)
img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# resize for display — keep it manageable
h, w = img_bgr.shape[:2]
MAX_DISPLAY = 700
scale = min(MAX_DISPLAY / w, MAX_DISPLAY / h, 1.0)
disp_w = int(w * scale)
disp_h = int(h * scale)
img_disp = cv2.resize(img_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
img_rgb  = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)

st.markdown("### Draw a rectangle around your subject")
st.caption("Click and drag to draw. Try to include a little margin around the subject. Redraw to adjust.")

canvas_result = st_canvas(
    fill_color="rgba(124, 111, 247, 0.15)",
    stroke_width=2,
    stroke_color="#7c6ff7",
    background_image=Image.fromarray(img_rgb),
    update_streamlit=True,
    width=disp_w,
    height=disp_h,
    drawing_mode="rect",
    key="canvas",
)

run = st.button("Remove Background", type="primary")

if not run:
    st.stop()

# ── extract rectangle from canvas ────────────────────────────────────────────
if canvas_result.json_data is None or not canvas_result.json_data.get("objects"):
    st.warning("Please draw a rectangle on the image first.")
    st.stop()

obj = canvas_result.json_data["objects"][-1]  # use last drawn rect

# canvas gives left, top, width, height in display coords
cx = int(obj["left"])
cy = int(obj["top"])
cw = int(obj["width"])
ch = int(obj["height"])

if cw < 10 or ch < 10:
    st.warning("Rectangle too small — draw a larger box around your subject.")
    st.stop()

# scale back to full resolution
x1 = max(0, int(cx / scale))
y1 = max(0, int(cy / scale))
x2 = min(w, int((cx + cw) / scale))
y2 = min(h, int((cy + ch) / scale))
rect_w = x2 - x1
rect_h = y2 - y1

# ── run GrabCut ───────────────────────────────────────────────────────────────
with st.spinner("Running GrabCut segmentation…"):
    mask      = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(img_bgr, mask, (x1, y1, rect_w, rect_h),
                    bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    except Exception as e:
        st.error(f"GrabCut failed: {e}")
        st.stop()

    fg_mask = np.where(
        (mask == GC_FGD) | (mask == GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # mask visualisation
    mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
    mask_vis[mask == GC_FGD]    = [80,  220, 80]
    mask_vis[mask == GC_PR_FGD] = [180, 220, 180]
    mask_vis[mask == GC_PR_BGD] = [80,  80,  100]
    mask_vis[mask == GC_BGD]    = [20,  20,  30]

    # checkerboard transparency preview
    board = np.zeros((h, w, 3), dtype=np.uint8)
    sz = 16
    for y in range(0, h, sz):
        for x in range(0, w, sz):
            val = 200 if (x//sz + y//sz) % 2 == 0 else 230
            board[y:y+sz, x:x+sz] = val

    fg_3ch = cv2.merge([fg_mask, fg_mask, fg_mask])
    img_rgb_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cutout = np.where(fg_3ch == 255, img_rgb_full, board)

    # transparent PNG
    bgra        = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:,:,3] = fg_mask

# ── results ───────────────────────────────────────────────────────────────────
st.markdown("### Results")
c1, c2, c3 = st.columns(3)

with c1:
    st.image(img_rgb_full, caption="Original", use_container_width=True)
with c2:
    st.image(mask_vis,
             caption="Segmentation mask — bright green=definite FG, light=probable FG, dark=BG",
             use_container_width=True)
with c3:
    st.image(cutout,
             caption="Cutout (checkerboard = transparent)",
             use_container_width=True)

fg_pct = int(np.sum(fg_mask > 0) / fg_mask.size * 100)
st.info(f"Foreground: {fg_pct}% of image  |  {np.sum(fg_mask > 0):,} pixels kept")

# ── download ──────────────────────────────────────────────────────────────────
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

st.caption("Tip: if the result is rough, increase GrabCut Iterations in the sidebar and click Remove Background again.")
