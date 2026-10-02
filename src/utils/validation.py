"""
Image conversion and dimension validation utilities.
"""
from typing import Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import io

from src.utils.io import get_logger

logger = get_logger(__name__)

def convert_uploaded_file_to_cv2(
    file_bytes_or_buffer: Union[bytes, io.BytesIO, any]
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Safely convert an uploaded file (from Streamlit st.file_uploader, BytesIO, or raw bytes)
    into a grayscale OpenCV uint8 numpy array (H, W).
    Handles RGB, RGBA, Grayscale, PNG, BMP, JPG, JPEG, corrupted, and empty files.

    Returns:
        (image_gray_array, error_message)
    """
    try:
        if file_bytes_or_buffer is None:
            return None, "No file provided."

        if hasattr(file_bytes_or_buffer, "read"):
            data = file_bytes_or_buffer.read()
        elif isinstance(file_bytes_or_buffer, bytes):
            data = file_bytes_or_buffer
        else:
            return None, f"Unsupported input type: {type(file_bytes_or_buffer)}"

        if len(data) == 0:
            return None, "Uploaded file is empty (0 bytes)."

        # First attempt decoding via OpenCV from memory buffer
        np_buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(np_buf, cv2.IMREAD_UNCHANGED)

        # Fallback to PIL if OpenCV cannot decode directly
        if img is None:
            try:
                pil_img = Image.open(io.BytesIO(data))
                img = np.array(pil_img)
            except Exception as pil_err:
                logger.error(f"Failed to decode image with OpenCV and PIL: {pil_err}")
                return None, f"Corrupted or invalid image file. Decoding failed: {pil_err}"

        if img is None or img.size == 0:
            return None, "Image contains zero pixels or could not be decoded."

        # Convert to Grayscale appropriately
        if len(img.shape) == 2:
            # Already single-channel grayscale
            img_gray = img
        elif len(img.shape) == 3:
            c = img.shape[2]
            if c == 3:
                # Assuming standard BGR from cv2.imdecode; if from PIL, RGB.
                # cv2.cvtColor with BGR2GRAY works well for biometric processing
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif c == 4:
                # RGBA -> BGR then GRAY (or directly BGRA2GRAY)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            elif c == 1:
                img_gray = img[:, :, 0]
            else:
                return None, f"Unsupported channel depth ({c} channels)."
        else:
            return None, f"Unusual image shape: {img.shape}"

        # Ensure uint8
        if img_gray.dtype != np.uint8:
            img_gray = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if img_gray.shape[0] < 10 or img_gray.shape[1] < 10:
            return None, f"Image resolution is too small ({img_gray.shape[1]}x{img_gray.shape[0]})."

        return img_gray, None

    except Exception as e:
        logger.error(f"Image conversion error: {e}", exc_info=True)
        return None, f"Error processing image: {str(e)}"

def validate_feature_dimension(
    features: np.ndarray,
    expected_dim: int,
    feature_name: str = "Feature vector"
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a feature vector matches the expected dimension.
    """
    if features is None:
        return False, f"{feature_name} is None."
    
    actual_dim = features.shape[-1] if len(features.shape) > 1 else len(features)
    if actual_dim != expected_dim:
        msg = f"{feature_name} dimension mismatch! Expected {expected_dim}, but got {actual_dim}."
        logger.error(msg)
        return False, msg
    return True, None
