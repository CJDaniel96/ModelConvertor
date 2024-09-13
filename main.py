import streamlit as st
import os
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from yolov5 import export


DIR = Path(__file__).absolute().parent
TEMP = DIR / "temp"
TEMP.mkdir(exist_ok=True)

@st.cache_data
def export_formats():
    # YOLOv5 export formats
    x = [
        ['PyTorch', '-', '.pt', True, True],
        ['TorchScript', 'torchscript', '.torchscript', True, True],
        ['ONNX', 'onnx', '.onnx', True, True],
        ['OpenVINO', 'openvino', '_openvino_model', True, False],
        ['TensorRT', 'engine', '.engine', False, True],
        ['CoreML', 'coreml', '.mlmodel', True, False],
        ['TensorFlow SavedModel', 'saved_model', '_saved_model', True, True],
        ['TensorFlow GraphDef', 'pb', '.pb', True, True],
        ['TensorFlow Lite', 'tflite', '.tflite', True, False],
        ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', False, False],
        ['TensorFlow.js', 'tfjs', '_web_model', False, False],
        ['PaddlePaddle', 'paddle', '_paddle_model', True, True], ]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])

def upload_file():
    st.subheader("Model Conversion")
    export_formats_df = export_formats()
    st.dataframe(export_formats_df, use_container_width=True)
    uploaded_file = st.file_uploader("Choose a file")
    convert_mode = st.selectbox("Select Convert Type", export_formats_df["Format"].tolist())
    model_quantization = st.selectbox("Select Model Quantization", ["FP32", "FP16", "INT8"])
    
    if uploaded_file and st.button("Start Model Conversion"):
        convert_argument = export_formats_df[export_formats_df["Format"] == convert_mode]["Argument"].values[0]
        convert_suffix = export_formats_df[export_formats_df["Format"] == convert_mode]["Suffix"].values[0]
        half, int8, dynamic, batch = get_quantization(model_quantization)
        model_converter(uploaded_file, convert_mode, convert_argument, convert_suffix, half, int8, dynamic, batch)

def get_quantization(model_quantization: str) -> tuple[bool, bool, bool, int]:
    if model_quantization == "FP32":
        half = False
        int8 = False
        dynamic = False
        batch = 1
    elif model_quantization == "FP16":
        half = True
        int8 = False
        dynamic = False
        batch = 1
    elif model_quantization == "INT8":
        half = False
        int8 = True
        dynamic = True
        batch = 8
    else:
        raise TypeError("Error Model Quantization")
    return half, int8, dynamic, batch

def get_file(file_path: Path, convert_suffix: str) -> Path:
    return file_path.with_suffix(convert_suffix)

def model_converter(uploaded_file, convert_mode: str, convert_argument: str, convert_suffix: str, half: bool, int8: bool, dynamic: bool, batch: int = 1):
    temp_file = TEMP.joinpath(uploaded_file.name)
    with open(str(temp_file), "wb") as f:
        f.write(uploaded_file.getvalue())
    try:
        model = YOLO(str(temp_file))
        model.export(format=convert_argument, half=half, int8=int8, dynamic=dynamic, batch=batch)
    except TypeError as e:
        export.run(weights=temp_file, include=[convert_argument], half=half, int8=int8, dynamic=dynamic, batch_size=batch)
    except Exception as e:
        st.write(f"**Error**: {e}")
        
    st.write("#### Results:")
    st.write(f"Model **{uploaded_file.name}** has been converted to **{convert_mode}** format.")
    os.remove(str(temp_file))
    dst_file = get_file(temp_file, convert_suffix)
    st.write(f"Save model to: **{str(dst_file)}**")
        
    with open(str(dst_file), "rb") as f:
        contents = f.read()
    st.download_button(label="Download Model", data=contents, file_name=dst_file.name)


def main():
    pages = {
        "Toolkits": [
            st.Page(upload_file, title="Model Convertor")
        ]
    }
    pg = st.navigation(pages)
    pg.run()


if __name__ == "__main__":
    main()