# mini_project

file structure:

bee-anomaly-detection/
├── backend/
│   ├── app.py             # Flask backend for API
│   ├── model/
│   │   ├── bee_model.h5   # Trained model
│   │   └── preprocess.py  # Preprocessing utilities
│   ├── uploads/           # Folder to store uploaded images
│   └── requirements.txt   # Backend dependencies
├── dataset/
│   ├── train/             # Training images
│   │   ├── healthy/
│   │   └── unhealthy/
│   └── val/               # Validation images
├── frontend/
│   ├── index.html         # Frontend HTML page
│   ├── style.css          # Frontend CSS for styling
│   └── script.js          # Frontend JavaScript for interaction
└── train_model.py         # Script for training the model


1. Training the Model
Create a Python script, train_model.py, to train the model using the VarroaDataset.

python
Copy code
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Paths
train_dir = 'dataset/train'
val_dir = 'dataset/val'

# Data augmentation and generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Model setup
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Save the model
model.save('backend/model/bee_model.h5')
print("Model saved successfully!")
2. Backend API (Flask)
Create backend/app.py to expose the model as a REST API.

python
Copy code
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import os
import numpy as np
from PIL import Image

app = Flask(__name__)
model_path = "backend/model/bee_model.h5"
model = load_model(model_path)

UPLOAD_FOLDER = 'backend/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    img = preprocess_image(file_path)
    prediction = model.predict(img)
    os.remove(file_path)

    result = 'Healthy' if prediction[0][0] < 0.5 else 'Unhealthy'
    return jsonify({'prediction': result})

if __name__ == "__main__":
    app.run(debug=True)
3. Frontend
frontend/index.html
html
Copy code
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="style.css">
    <title>Bee Anomaly Detection</title>
</head>
<body>
    <h1>Bee Anomaly Detection</h1>
    <form id="uploadForm">
        <input type="file" id="fileInput" name="file" accept="image/*" required>
        <button type="submit">Upload and Predict</button>
    </form>
    <p id="result"></p>
    <script src="script.js"></script>
</body>
</html>
frontend/style.css
css
Copy code
body {
    font-family: Arial, sans-serif;
    text-align: center;
    margin: 0;
    padding: 0;
    background-color: #f4f4f4;
}
h1 {
    margin: 20px;
}
form {
    margin-top: 50px;
}
#result {
    margin-top: 20px;
    font-size: 18px;
    font-weight: bold;
}
frontend/script.js
javascript
Copy code
document.getElementById('uploadForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (!file) {
        alert("Please upload a file.");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    document.getElementById('result').textContent = `Prediction: ${result.prediction}`;
});
4. Dependencies
Create backend/requirements.txt for backend dependencies:

makefile
Copy code
Flask==2.3.3
tensorflow==2.13.0
numpy==1.24.3
Pillow==9.5.0
Next Steps
Train the Model: Run train_model.py to create bee_model.h5.
Run Backend: Start the Flask server using:
bash
Copy code
python backend/app.py
Open Frontend: Load frontend/index.html in your browser.
Test Prediction: Upload a bee image and check the prediction.
