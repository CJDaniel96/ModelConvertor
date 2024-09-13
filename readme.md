# Model Conversion Tool

This project provides a web-based tool for converting machine learning models into different formats using Streamlit. The tool supports various conversion formats and model quantization options.

## Features

- Upload a machine learning model file.
- Select the desired conversion format.
- Choose the model quantization type.
- Convert the model and download the converted file.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/model-conversion-tool.git
    cd model-conversion-tool
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On macOS/Linux
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run main.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Use the web interface to upload your model file, select the conversion format, and choose the model quantization type.

4. Click the "Start Model Conversion" button to convert the model.

5. Download the converted model using the provided download button.

## Code Overview

### Main Functions

- `upload_file()`: Handles the file upload and conversion process.
- `export_formats()`: Returns a DataFrame containing the supported export formats and their properties.
- `get_quantization(model_quantization: str) -> tuple[bool, bool, bool, int]`: Determines the quantization settings based on the selected model quantization type.
- `get_file(file_path: Path, convert_suffix: str) -> Path`: Generates the output file path with the appropriate suffix.
- `model_converter(uploaded_file, convert_mode: str, convert_argument: str, convert_suffix: str, half: bool, int8: bool, dynamic: bool, batch: int = 1)`: Converts the uploaded model file using the specified conversion settings.

### JavaScript Integration

The application includes a JavaScript snippet to detect when the page is closed and stop the Streamlit service.

### Flask Integration

A Flask server is used to handle the shutdown request triggered by the JavaScript snippet.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.