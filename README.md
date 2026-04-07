# Background Remover (Web)

A browser-based background removal tool built with Streamlit, using the GrabCut algorithm.

Upload an image, use the sliders to position a rectangle around your subject, and download the result as a transparent PNG.

## Live Demo
[bgremover-kasvi.streamlit.app](https://bgremover-kasvi.streamlit.app)

## Run locally
pip install streamlit opencv-python numpy Pillow
streamlit run app.py

## Note on quality
The desktop version of this project used mouse-drawn rectangles directly on the image for precise initialisation. In the web version, sliders are used instead — this makes it harder to tightly crop the subject, which directly affects GrabCut's accuracy since it relies heavily on a precise initialisation rectangle. We attempted to use `streamlit-drawable-canvas` for mouse drawing but it is currently incompatible with the Streamlit Cloud Python environment. The core CV algorithm (GrabCut) is unchanged — the limitation is purely in the interface.
