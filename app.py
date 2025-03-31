import os
from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__, static_folder='styles', template_folder='.')
CORS(app)

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
DATASET_FOLDER = 'datasets'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
models = {
    'Rice': load_model(os.path.join(MODEL_FOLDER, 'rice_resnet50_finetuned.h5')),
    'Potato': load_model(os.path.join(MODEL_FOLDER, 'potato_resnet50_finetuned.h5')),
    'Wheat': load_model(os.path.join(MODEL_FOLDER, 'wheat2_resnet50_finetuned.h5')),
    'Cauliflower': load_model(os.path.join(MODEL_FOLDER, 'cauliflower_resnet50_finetuned.h5')),
    'Tomato': load_model(os.path.join(MODEL_FOLDER, 'tomato_resnet50_finetuned.h5'))

}

# Dynamically get class labels from dataset folders
def get_classes(dataset_name):
    dataset_path = os.path.join(DATASET_FOLDER, dataset_name)
    return sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

dataset_classes = {
    'Rice': get_classes('good_dataset'),
    'Potato': get_classes('potato_dataset'),
    'Wheat': get_classes('Wheat_leaf_dataset'),
    'Cauliflower': get_classes('cauliflower_dataset'),
    'Tomato': get_classes('tomato_dataset1')

}


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, (300, 300))
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)  
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['image']
        model_type = request.form.get('model', 'Rice')  # Default to 'Rice'
        
        if model_type not in models:
            return jsonify({'error': 'Invalid model selection'}), 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        img = preprocess_image(file_path)

        prediction = models[model_type].predict(img)[0]
        predicted_class = np.argmax(prediction)
        category = dataset_classes[model_type][predicted_class]

        print(f"Prediction: {prediction}")
        print(f"Predicted Class Index: {predicted_class}")
        print(f"Class Mapping: {dataset_classes[model_type]}")


        os.remove(file_path)
        return jsonify({'category': category, 'confidence': float(np.max(prediction))})


    except Exception as e:
        app.logger.error(f"Error in classification: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)


# import os
# from flask import Flask, request, jsonify, render_template
# import numpy as np
# import cv2
# from flask_cors import CORS
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.resnet50 import preprocess_input

# app = Flask(__name__, static_folder='styles', template_folder='.')
# CORS(app)

# UPLOAD_FOLDER = 'uploads'
# MODEL_FOLDER = 'models'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Load models
# models = {
#     'Rice': load_model(os.path.join(MODEL_FOLDER, 'rice_resnet50_finetuned.h5')),
#     'Potato': load_model(os.path.join(MODEL_FOLDER, 'potato_resnet50_finetuned.h5')),
#     'Wheat': load_model(os.path.join(MODEL_FOLDER, 'wheat2_resnet50_finetuned.h5')),
#     'Cauliflower': load_model(os.path.join(MODEL_FOLDER, 'cauliflower_resnet50_finetuned.h5')),
#     'Tomato': load_model(os.path.join(MODEL_FOLDER, 'tomato_resnet50_finetuned.h5'))
# }

# # Manually define dataset classes
# dataset_classes = {
#     'Rice': ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy Rice Leaf', 'Leaf Blast', 'Leaf Scald', 'Narrow Leaf Brown Spot', 'Rice Hispa', 'Sheath Blight'],
#     'Potato': ['Bacteria', 'Fungi', 'Healthy', 'Nematode', 'Pest', 'Phytopthora', 'Virus'],
#     'Wheat': ['Black Rust', 'Blast', 'Brown Rust', 'Fusarium Head Blight', 'Healthy', 'Leaf Blight', 'Mildew', 'Septoria', 'Smut', 'Tan Spot', 'Yellow Rust'],
#     'Cauliflower': ['Bacterial Spot Rot', 'Black Rot', 'Downy Mildew', 'No Disease'],
#     'Tomato': ['Bacterial Spot', 'Early Blight', 'Healthy', 'Late Blight', 'Leaf Mold', 'Septoria Leaf Spot', 'Spider Mites Two Spotted Spider Mite', 'Target Spot', 'Tomato Mosaic Virus', 'Tomato Yellow Leaf Curl Virus']
# }


# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     if img is not None:
#         img = cv2.resize(img, (300, 300))
#         img = preprocess_input(img)
#         img = np.expand_dims(img, axis=0)  
#     return img

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/classify', methods=['POST'])
# def classify_image():
#     try:
#         if 'image' not in request.files:
#             return jsonify({'error': 'No file uploaded'}), 400

#         file = request.files['image']
#         model_type = request.form.get('model', 'Rice')  # Default to 'Rice'
        
#         if model_type not in models:
#             return jsonify({'error': 'Invalid model selection'}), 400

#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(file_path)

#         img = preprocess_image(file_path)

#         prediction = models[model_type].predict(img)[0]
#         predicted_class = np.argmax(prediction)
#         category = dataset_classes[model_type][predicted_class]

#         print(f"Prediction: {prediction}")
#         print(f"Predicted Class Index: {predicted_class}")
#         print(f"Class Mapping: {dataset_classes[model_type]}")

#         os.remove(file_path)
#         return jsonify({'category': category, 'confidence': float(np.max(prediction))})

#     except Exception as e:
#         app.logger.error(f"Error in classification: {e}")
#         return jsonify({'error': 'Internal server error'}), 500

# if __name__ == '__main__':
#     app.run(debug=False, host='0.0.0.0', port=5000)
