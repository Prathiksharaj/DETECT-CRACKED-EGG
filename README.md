# DETECT-CRACKED-EGG
This system is used to detect cracked eggs and differentiate between  healthy, good and impure eggs on a conveyor belt.
📌 Project Overview

This project consists of two major functionalities:

Cracked Egg Detection: Detects and classifies eggs into three categories:

Good Eggs

Medium Impurity Eggs

Cracked Eggs

Egg Classification: Uses a deep learning model to classify eggs as either Chicken or Duck eggs.

🛠️ Tech Stack

Backend: Flask

Machine Learning: TensorFlow/Keras

Image Processing: OpenCV, NumPy

Frontend: HTML, CSS (Jinja2 templates in Flask)

🚀 Features

🥚 Cracked Egg Detection

Uses OpenCV to detect eggs in an image using Hough Circle Transform.

Applies edge detection and thresholding for abnormality detection.

Classifies eggs into three categories based on impurity levels.

Outputs an image with bounding boxes and classification results.

🐥 Egg Classification (Chicken vs. Duck)

Uses a pre-trained deep learning model (my_model.keras).

Loads an image, preprocesses it, and predicts whether the egg belongs to a Chicken or a Duck.

Outputs the classification results along with the original images.

📂 Project Structure

├── backend/
│   ├── app.py                  # Flask application
│   ├── detection.py            # Cracked egg detection logic (OpenCV)
│   ├── templates/
│   │   ├── upload.html         # Upload page for egg detection
│   │   ├── results.html        # Results page for detected cracked eggs
│   │   ├── upload_predict.html # Upload page for classification
│   │   ├── predictions.html    # Results page for classification
│   ├── uploads/                # Uploaded images
│   ├── results/                # Processed images with bounding boxes
│   ├── classification_results/ # Classified images (Chicken/Duck)
├── my_model.keras              # Pre-trained Keras model
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies

⚙️ Setup & Installation



1 Install Dependencies

pip install Flask tensorflow numpy opencv-python

2 Run the Flask App

python app.py

3 Access the Web App

Open a browser and go to: http://127.0.0.1:5000/

🔬 How It Works

🥚 Cracked Egg Detection

Upload an image containing eggs.

The system detects cracked, medium impurity, and good eggs.

It marks the cracked eggs (blue box) and medium impurity eggs (yellow box).

The results are displayed and can be downloaded.

🐣 Egg Classification

Upload two images for classification.

The pre-trained model predicts whether the egg is Chicken or Duck.

Results are displayed with original images.


🎯 Future Enhancements

✅ Improve accuracy using a more robust deep learning model.
✅ Implement real-time detection using a camera feed.
✅ Add support for mobile uploads.

📜 License

This project is open-source and free to use.

👩‍💻 Author

Prathiksha 👩‍💻

For any queries, feel free to connect! 🚀

