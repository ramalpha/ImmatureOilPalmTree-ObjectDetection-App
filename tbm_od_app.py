import streamlit as st
from deepforest import main as deepforest_main
import cv2
import os
import tempfile
import torch
import rasterio
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from io import BytesIO

# --- Configuration ---
DEEPFOREST_MODEL_PATH = "models/tbmdetection-v9.pth"
DEMO_IMG_PATH = "demo/demo_image.PNG"

# --- Model Loading ---
@st.cache_resource
def load_deepforest_model(checkpoint_path):
    st.write(f"Attempting to load DeepForest model from: {checkpoint_path}")
    try:
        # !!! IMPORTANT: Customize num_classes and label_dict for the model !!!
        label_mapping = {"tbm-1": 0, "tbm-2": 1} 
        num_classes_for_model = 2
        
        model = deepforest_main.deepforest(
            num_classes=num_classes_for_model, 
            label_dict=label_mapping
        )
        st.write(f"Initializing DeepForest with {num_classes_for_model} classes and label_dict: {label_mapping}")
        
        model.model.load_state_dict(
            torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        )
        model.eval()
        st.success("DeepForest model loaded successfully with custom classes!")
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {checkpoint_path}. Please check the path.")
        return None
    except Exception as e:
        st.error(f"Error loading the DeepForest model: {str(e)}") 
        st.error(
            "Please ensure DeepForest is installed correctly, the model path is valid, "
            "and the model architecture (num_classes, label_dict) matches the loaded weights."
        )
        return None

# --- Main Prediction Function ---
def prediction_on_image(model, image_path, score_thresh=0.5,
                        save_dir=None, custom_filename=None,
                        patch_size=800,       
                        patch_overlap=0.1,   
                        min_tile_size=4000,   # Threshold to switch to predict_tile
                        max_pixels=178956970   
                       ):
    """
    Predicts objects in an image using the provided DeepForest model and visualizes the results.
    Handles multiple classes, assigns different colors, dynamically adjusts label text size.
    Uses predict_tile for large images and predict_image for smaller ones.
    Compresses the image if it exceeds the specified pixel limit during Pillow load.
    Returns: (visualized_image_numpy_array, prediction_result_dataframe)
    """
    model.config["score_thresh"] = score_thresh
    image_np = None
    image_width = 0
    image_height = 0
    
    original_max_image_pixels = Image.MAX_IMAGE_PIXELS
    try:
        # Temporarily increase Pillow's limit for its fallback path, 
        # relying on internal max_pixels to do the actual scaling.
        Image.MAX_IMAGE_PIXELS = 1000000000
        
        try: # 1. Open with rasterio if possible
            with rasterio.open(image_path) as src:
                image_data = src.read() 
                if image_data.ndim == 3 and image_data.shape[0] in (3, 4): 
                    image_np = image_data.transpose(1, 2, 0).copy() 
                elif image_data.ndim == 2: 
                    image_np = image_data.copy() 
                else: 
                    image_np = image_data.copy() 
                image_width, image_height = src.width, src.height
        except rasterio.errors.RasterioIOError: # 2. Fallback to pillow
            try:
                with Image.open(image_path) as image:
                    image_width_pil, image_height_pil = image.size # Use different vars for clarity before resize
                    if image_width_pil * image_height_pil > max_pixels: 
                        resize_factor = (max_pixels / (image_width_pil * image_height_pil))**0.5
                        new_width = int(image_width_pil * resize_factor)
                        new_height = int(image_height_pil * resize_factor)
                        st.info(f"Image too large, resizing to {new_width}x{new_height} pixels from {image_width_pil}x{image_height_pil}.")
                        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    image_np = np.array(image.convert("RGB")).copy() 
                    image_width, image_height = image.size # Final dimensions after potential resize
            except Exception as e:
                st.error(f"Error loading image with PIL: {e}")
                return None, pd.DataFrame() 
        
        if image_np is None:
            st.error(f"Could not load image: {image_path}")
            return None, pd.DataFrame()

    except (UnidentifiedImageError, FileNotFoundError) as e:
        st.error(f"Error loading image: {e}")
        return None, pd.DataFrame()
    finally:
        Image.MAX_IMAGE_PIXELS = original_max_image_pixels # Reset to Pillow's default

    try: # Ensure image_np is RGB (3 channels) for consistent processing
        if image_np.ndim == 3 and image_np.shape[2] == 4: 
            image_np = image_np[:, :, :3].copy()
        elif image_np.ndim == 2: 
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB).copy()
        elif image_np.ndim == 3 and image_np.shape[2] == 3:
            if not image_np.flags.owndata: 
                 image_np = image_np.copy()
        
        if image_np.ndim != 3 or image_np.shape[2] != 3:
             st.error(f"Image could not be converted to 3-channel RGB. Final shape: {image_np.shape}")
             return image_np, pd.DataFrame() 
    except Exception as e:
        st.error(f"Error processing image channels: {e}")
        return image_np, pd.DataFrame() # Return potentially original image_np for context if possible

    # Perform prediction
    if image_width > min_tile_size or image_height > min_tile_size:
        prediction_result = model.predict_tile(
            image=image_np,
            patch_size=patch_size, patch_overlap=patch_overlap
        )
    else:
        prediction_result = model.predict_image(image_np)

    if prediction_result is None: 
        prediction_result = pd.DataFrame()

    # Visualization
    output_image = image_np.copy() 
    if not prediction_result.empty and 'xmin' in prediction_result.columns:
        color_map = {"tbm-1": (0, 255, 255), "tbm-2": (204, 255, 0)} 
        default_color = (255, 0, 0) 

        for index, row in prediction_result.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = row['label']
            score = row['score']

            box_width = x2 - x1
            box_height = y2 - y1
            font_scale = max(0.3, min(box_width, box_height) / 150.0) 
            font_thickness = max(1, int(font_scale * 2))
            color = color_map.get(label, default_color)

            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output_image, f"{label} {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
    
    if save_dir: 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        base_filename = os.path.basename(image_path)
        name_part, ext_part = os.path.splitext(base_filename)
        filename_to_save = custom_filename if custom_filename else f"{name_part}_detected{ext_part}"
        save_path = os.path.join(save_dir, filename_to_save)
        try:
            if output_image.ndim == 3 and output_image.shape[2] == 3:
                 cv2.imwrite(save_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
            else: 
                 cv2.imwrite(save_path, output_image)
            print(f"Saved visualized image to {save_path}") 
        except Exception as e:
            print(f"Error saving image with cv2.imwrite: {e}") 
            
    return output_image, prediction_result

# --- Main Streamlit Application ---
st.set_page_config(layout="wide") 
st.title('Immature Oil Palm Detection')

# --- Load Model ---
model = load_deepforest_model(DEEPFOREST_MODEL_PATH)

# --- Sidebar ---
st.sidebar.title("Controls")
st.sidebar.markdown("---")
img_file_buffer = st.sidebar.file_uploader(
    "Upload an Image", 
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)
score_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01, key="score_slider")
patch_size_control = st.sidebar.number_input("Patch Size (for Tiling)", min_value=200, max_value=1000, value=800, step=100, key="patch_size_input")
st.sidebar.markdown("---")

# --- Main Panel ---
if model is not None:
    image_path_for_prediction = None
    uploaded_image_path_for_cleanup = None 

    if img_file_buffer is not None: 
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(img_file_buffer.name)[1]) as tmp_file:
            tmp_file.write(img_file_buffer.getvalue())
            image_path_for_prediction = tmp_file.name
            uploaded_image_path_for_cleanup = image_path_for_prediction
        
        st.sidebar.markdown("**Original Uploaded Image**")
        try:
            original_image_pil = Image.open(image_path_for_prediction)
            st.sidebar.image(original_image_pil, use_container_width=True) 
        except Exception as e:
            st.sidebar.error(f"Could not display uploaded image preview: {e}")
            image_path_for_prediction = None 
    else: 
        image_path_for_prediction = DEMO_IMG_PATH
        st.sidebar.markdown("**Original Image (Demo)**")
        original_pillow_max_pixels = Image.MAX_IMAGE_PIXELS 
        try:
            # Temporarily increase for known large demo TIFF if needed by Pillow
            # Ensure your DEMO_IMG_PATH is now a smaller image if possible
            if DEMO_IMG_PATH.lower().endswith((".tif", ".tiff")): # Only for TIFF demo
                 Image.MAX_IMAGE_PIXELS = 650000000 
            demo_image_pil = Image.open(DEMO_IMG_PATH)
            st.sidebar.image(demo_image_pil, use_container_width=True) 
            if 'demo_info_shown' not in st.session_state: 
                 st.info("This is a demo image. Upload your own image using the sidebar control.")
                 st.session_state['demo_info_shown'] = True
        except FileNotFoundError:
            st.sidebar.error(f"Demo image '{DEMO_IMG_PATH}' not found.")
            image_path_for_prediction = None
        except Exception as e:
            st.sidebar.error(f"Could not display demo image: {e}")
            image_path_for_prediction = None
        finally:
            Image.MAX_IMAGE_PIXELS = original_pillow_max_pixels # Reset to default

    if image_path_for_prediction: 
        st.header("Prediction Output")
        
        with st.spinner("Analyzing image... This might take a moment."):
            output_image_np, detections_df = prediction_on_image(
                model=model,
                image_path=image_path_for_prediction,
                score_thresh=score_threshold,
                patch_size=patch_size_control # Pass the value from the sidebar
            )

        if output_image_np is not None:
            st.subheader("Detection Summary")
            if detections_df is not None and not detections_df.empty:
                num_total_detections = len(detections_df)
                st.markdown(f"#### Total Detections Found: {num_total_detections}")

                class_counts_series = detections_df['label'].value_counts()
                st.markdown("##### Detections per Class:")
                class_counts_df_for_csv = class_counts_series.reset_index()
                class_counts_df_for_csv.columns = ['Class Label', 'Count']

                for class_label, count_val in class_counts_series.items():
                    st.markdown(f"- **{class_label}:** {count_val}")
            else: 
                st.markdown("#### No Detections Found")
                class_counts_df_for_csv = pd.DataFrame(columns=['Class Label', 'Count'])
            
            st.markdown("---")
            st.image(output_image_np, use_container_width=True, caption="Image with Detections") 
            st.markdown("---")

            st.subheader("Downloads")
            col1, col2 = st.columns(2)
            original_filename_base = os.path.splitext(os.path.basename(image_path_for_prediction))[0]

            try: 
                pil_img_to_download = Image.fromarray(output_image_np.astype(np.uint8))
                buf = BytesIO()
                pil_img_to_download.save(buf, format="PNG")
                image_bytes_to_download = buf.getvalue()
                with col1:
                    st.download_button(
                        label="Download Visualized Image (PNG)",
                        data=image_bytes_to_download,
                        file_name=f"detected_{original_filename_base}.png",
                        mime="image/png"
                    )
            except Exception as e:
                with col1:
                    st.error(f"Error preparing image for download: {str(e)[:200]}")

            try: 
                csv_data = class_counts_df_for_csv.to_csv(index=False).encode('utf-8')
                with col2:
                    st.download_button(
                        label="Download Detection Counts (CSV)",
                        data=csv_data,
                        file_name=f"detection_counts_{original_filename_base}.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                with col2:
                    st.error(f"Error preparing CSV for download: {str(e)[:200]}")
        else: 
            st.error("Failed to process the image: No output image was generated.")
    else: 
        if img_file_buffer is None and (not DEMO_IMG_PATH or not os.path.exists(DEMO_IMG_PATH)):
            st.warning("Please upload an image, or ensure the demo image path is correctly configured.")

    if uploaded_image_path_for_cleanup and os.path.exists(uploaded_image_path_for_cleanup):
        try:
            os.remove(uploaded_image_path_for_cleanup)
        except Exception: 
            pass 
else: 
    st.header("DeepForest Model could not be loaded.")
    st.markdown(f"""
    Please ensure:
    1. Your model file is correctly specified at `{DEEPFOREST_MODEL_PATH}`.
    2. DeepForest and all dependencies are installed (check `requirements.txt`).
    3. The `load_deepforest_model` function correctly initializes `deepforest_main.deepforest()` 
       with `num_classes` and `label_dict` matching your trained model.
    """)

# --- Footer Section ---
st.markdown("---")
footer_html = """
<div style="text-align: center; margin-top: 2em; line-height: 1.8;">
    <p style="margin-bottom: 0.5em;">App by <a href="https://github.com/ramalpha" target="_blank" style="text-decoration: none; color: #0366d6;">Ramadya Alif Satya</a></p>
    <p style="margin-top: 0.2em; margin-bottom: 0.5em;">
        <a href="https://github.com/ramalpha" target="_blank">
            <img alt="GitHub Profile" src="https://img.shields.io/badge/GitHub-ramalpha-blue?logo=github&style=for-the-badge">
        </a>
    </p>
    <p style="margin-top: 0.2em;">Based on DeepForest and Streamlit</p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)