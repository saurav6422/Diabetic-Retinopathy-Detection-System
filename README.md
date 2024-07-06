# Diabetic Retinopathy Detection System

This repository contains a web application built using Flask and HTML/CSS for detecting different stages of diabetic retinopathy from retinal images. 
The application allows users to upload scanned retinal images and provides predictions on the stage of diabetic retinopathy, including mild, moderate, no diabetic retinopathy, proliferative diabetic retinopathy, and severe diabetic retinopathy.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Team](#team)

## Features

- User authentication: Secure login page to access the application.
- Image upload: Users can upload scanned retinal images for analysis.
- Diabetic retinopathy prediction: The application utilizes a pre-trained model to predict the stage of diabetic retinopathy based on the uploaded image.
- Result display: The application displays the predicted stage of diabetic retinopathy along with relevant information and recommendations.
- About and contact pages: Additional pages provide information about the application and contact details.

## Requirements

- Python 3.7+
- [requirements.txt](requirements.txt)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/saurav6422/Diabetic-Retinopathy-Detection-System.git
    cd Diabetic-Retinopathy-Detection-System
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:
    ```bash
    flask run 
    ```

2. Open your browser and go to `[http://localhost:5000]` to access the application.
3. Log in to the application using the provided credentials.
4. Navigate to the home page and click the "Upload Image" button.
5. Select a scanned retinal image from your local filesystem.
6. Wait for the application to process the image and display the predicted stage of diabetic retinopathy.
7. Follow the provided recommendations or seek medical advice based on the prediction.

## File Structure

- `app.py`: The main script containing the Streamlit application.
- `requirements.txt`: A file listing the required Python packages.
- `model2_.h5` : Trained model.
- `static` : Assets for the website.
- `templates` : File for the website.
- `dataset` : dataset upon which model is trained.

## License

This project is licensed under the GNU General Public License v2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Flask](https://palletsprojects.com/p/flask/)
- [OpenCV](https://opencv.org/)
- [Keras](https://keras.io/)
- [TenserFlow](https://www.tensorflow.org/)

## Team

- Aditya - Machine Learning Engineer - [Git](https://github.com/Aditya-039)
- Saurav - Machine Learning Engineer | Backend Developer - [Git](https://github.com/saurav6422)
- Sambarta - Frontend Developer - [Git](https://github.com/Sambarta-2001)
- Abir - Frontend Developer - [Git](https://github.com/abir-011)

