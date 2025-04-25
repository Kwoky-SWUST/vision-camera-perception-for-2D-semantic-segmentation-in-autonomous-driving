
# vision-camera-perception-for-2D-semantic-segmentation-in-autonomous-driving

## Overview

This repository contains code and resources related to vision-based camera perception for 2D semantic segmentation in the context of autonomous driving.

## Installation

### Prerequisites

*   Python 3.6+
*   PyTorch (version compatible with your hardware)
*   Other dependencies (install using `pip install -r requirements.txt` after creating it based on the imports in the python files)

### Steps

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd vision-camera-perception-for-2D-semantic-segmentation-in-autonomous-driving
    ```

2.  Install the required Python packages.  Since a `requirements.txt` file is not provided, you'll need to create one based on the imports in the Python files (`colormap.py`, `labelgenerator.py`, `predictor.py`).  For example:

    ```bash
    # Example requirements.txt (adjust based on your actual dependencies)
    torch
    torchvision
    pillow  # For image handling
    numpy
    # Add any other libraries used in the Python files
    ```

    Then, install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  Download the pre-trained model:

    The repository includes a pre-trained model `semantic_segmentation_mlp.pth`. Ensure it is located in the root directory of the project.

## Key Features

*   **Semantic Segmentation:**  Performs pixel-wise classification of images, assigning each pixel to a specific semantic category (e.g., road, car, pedestrian).
*   **Pre-trained Model:** Includes a pre-trained model (`semantic_segmentation_mlp.pth`) for immediate use.
*   **Data Handling:** Contains scripts for generating labels (`labelgenerator.py`) and applying colormaps (`colormap.py`).
*   **Prediction Script:** Provides a `predictor.py` script for running inference on images.
*   **Example Images:** Includes example training and testing images (`train_image.png`, `test_image.png`) and their corresponding overlays and pseudo-labels.
*   **Task Description:** Includes a PDF document (`Task_PhD_position.pdf`) outlining the project's goals and context.

## API Documentation

While detailed API documentation is not explicitly provided, the core functionality is exposed through the `predictor.py` script.

### `predictor.py`

This script likely contains functions for:

*   Loading the pre-trained model.
*   Preprocessing input images.
*   Performing semantic segmentation.
*   Post-processing the segmentation results (e.g., applying colormaps).

Inspect the script for specific function signatures and usage examples.

## Contribution Guidelines

Contributions are welcome! To contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Implement your changes.
4.  Test your changes thoroughly.
5.  Submit a pull request.

Please ensure your code adheres to the project's coding style and includes appropriate documentation.

## Troubleshooting

*   **CUDA Errors:** If you encounter CUDA-related errors, ensure that you have the correct CUDA drivers and PyTorch version installed for your GPU.
*   **Missing Dependencies:** Double-check that you have installed all the required Python packages using `pip install -r requirements.txt`.  If `requirements.txt` is missing, create it based on the imports in the python files.
*   **Model Loading Errors:** Verify that the `semantic_segmentation_mlp.pth` file is present in the correct directory and that the path to the model is correctly specified in the `predictor.py` script.
*   **Image Size Mismatch:** Ensure that the input images are of the expected size for the model.  Check the `predictor.py` script for any image resizing or cropping operations.

## License

License information is not provided.  Please contact the owner (Kwoky-SWUST) for licensing details.
```
