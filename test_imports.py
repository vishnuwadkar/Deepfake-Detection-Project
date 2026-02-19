import tensorflow as tf
import cv2
import mtcnn
import streamlit
import matplotlib
import matplotlib.pyplot as plt

try:
    print(f"TensorFlow version: {tf.__version__}")
except Exception as e:
    print(f"TensorFlow import failed: {e}")

try:
    print(f"OpenCV version: {cv2.__version__}")
except Exception as e:
    print(f"OpenCV import failed: {e}")

try:
    print(f"MTCNN version: {mtcnn.__version__}")
except Exception as e:
    print(f"MTCNN import failed: {e}")

try:
    print(f"Streamlit version: {streamlit.__version__}")
except Exception as e:
    print(f"Streamlit import failed: {e}")

try:
    print(f"Matplotlib version: {matplotlib.__version__}")
except Exception as e:
    print(f"Matplotlib import failed: {e}")

print("Import verification script completed.")
