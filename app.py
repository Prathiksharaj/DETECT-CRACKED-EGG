from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from detection import process_image  # Custom image processing function
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['CLASSIFICATION_RESULT_FOLDER'] = 'classification_results'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(app.config['CLASSIFICATION_RESULT_FOLDER'], exist_ok=True)

# Load the model for predicting egg types
loaded_model = load_model('my_model.keras')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process the image for cracked eggs
            result_image, cracked_count, medium_impurity_count, good_count = process_image(file_path)
            result_image_path = os.path.join(app.config['RESULT_FOLDER'], filename)
            cv2.imwrite(result_image_path, result_image)
            
            # Render results template
            return render_template(
                'results.html',
                cracked_count=cracked_count,
                medium_impurity_count=medium_impurity_count,
                good_count=good_count,
                filename=filename
            )
    return render_template('upload.html')

@app.route('/upload_predict', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        if 'file1' not in request.files or 'file2' not in request.files:
            return 'Both files are required'
        
        file1 = request.files['file1']
        file2 = request.files['file2']

        if file1.filename == '' or file2.filename == '':
            return 'No selected file'

        # Save files
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)

        file_path1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        file_path2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

        file1.save(file_path1)
        file2.save(file_path2)

        # Predict using the loaded model
        prediction1 = predict_image(file_path1)
        prediction2 = predict_image(file_path2)

        # Save classification results to a different folder
        classification_image_path1 = os.path.join(app.config['CLASSIFICATION_RESULT_FOLDER'], filename1)
        classification_image_path2 = os.path.join(app.config['CLASSIFICATION_RESULT_FOLDER'], filename2)

        # Optionally save the original images or processed images here
        cv2.imwrite(classification_image_path1, cv2.imread(file_path1))  # Example for saving original image
        cv2.imwrite(classification_image_path2, cv2.imread(file_path2))  # Example for saving original image

        # Render predictions
        return render_template('predictions.html', 
                               prediction1=prediction1, 
                               prediction2=prediction2, 
                               filename1=filename1, 
                               filename2=filename2)

    return render_template('upload_predict.html')

def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (150, 150))
    img = img.astype('float32') / 255.0
    img_array = np.expand_dims(img, axis=0)

    prediction = loaded_model.predict(img_array)
    return 'Duck' if prediction[0] < 0.5 else 'Chicken'

@app.route('/result/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename))

@app.route('/classification/<filename>')
def classification_file(filename):
    return send_file(os.path.join(app.config['CLASSIFICATION_RESULT_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True)
