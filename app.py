import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adamax
from flask import Flask, render_template, request
from PIL import Image
import cv2
import sys
from io import StringIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the trained model
loaded_model = tf.keras.models.load_model('models/BT_GC_Stage2.h5', compile=False)
loaded_model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def generate_heatmap(model, img_array):
    last_conv_layer = model.get_layer('Top_Conv_Layer')
    heatmap_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = heatmap_model(img_array)
        if predictions.shape[1] > 1:
            pred_index = tf.argmax(predictions[0])
        else:
            pred_index = 0
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap[0]

def apply_heatmap(image_path, heatmap):
    img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    superimposed_img_path = f"static/images/superimposed_{os.path.basename(image_path)}"
    cv2.imwrite(superimposed_img_path, superimposed_img)

    return superimposed_img_path

def predict(image_path):
    img_array = preprocess_image(image_path)
    predictions = loaded_model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_class_index]
    heatmap = generate_heatmap(loaded_model, img_array)
    heatmap_image_path = apply_heatmap(image_path, heatmap)

    # Additional information for each class
    additional_info = {
    'Glioma': {
        'description': 'Gliomas are a type of tumor that occurs in the brain and spinal cord. They can be quite serious and require prompt treatment.',
        'references': [
            {
                'title': 'Glioma - Mayo Clinic',
                'link': 'https://www.mayoclinic.org/diseases-conditions/glioma/symptoms-causes/syc-20350251'
            },
            {
                'title': 'Glioma - American Brain Tumor Association',
                'link': 'https://www.abta.org/tumor_types/glioma/'
            }
        ]
    },
    'Meningioma': {
        'description': 'Meningiomas are generally slow-growing tumors that form in the meninges, the protective layers of the brain and spinal cord.',
        'references': [
            {
                'title': 'Meningioma - Johns Hopkins Medicine',
                'link': 'https://www.hopkinsmedicine.org/health/conditions-and-diseases/meningioma'
            },
            {
                'title': 'Meningioma - American Association of Neurological Surgeons',
                'link': 'https://www.aans.org/en/Patients/Neurosurgical-Conditions-and-Treatments/Meningioma'
            }
        ]
    },
    'No Tumor': {
        'description': 'No tumor detected. The image appears to be of a healthy brain.',
        'references': [
            {
                'title': 'Normal Brain MRI - Radiopaedia',
                'link': 'https://radiopaedia.org/articles/normal-brain-mri-1?lang=us'
            }
        ]
    },
    'Pituitary': {
        'description': 'Pituitary tumors are growths that occur in the pituitary gland. They can affect hormone levels and may require treatment.',
        'references': [
            {
                'title': 'Pituitary Tumors - American Cancer Society',
                'link': 'https://www.cancer.org/cancer/pituitary-tumors.html'
            },
            {
                'title': 'Pituitary Adenomas - The Pituitary Society',
                'link': 'https://pituitarysociety.org/patient-education/pituitary-disorders/pituitary-tumors/pituitary-adenomas'
            }
        ]
    }
}


    additional_info_for_class = additional_info[class_labels[predicted_class_index]]

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

     # Log the steps of the heatmap algorithm
    input_img=preprocess_image(image_path)
    model=loaded_model
    all_layer_outputs = []
    for layer in model.layers:
        try:
            intermediate_model = tf.keras.models.Model(inputs=model.inputs, outputs=layer.output)
            layer_output = intermediate_model.predict(input_img)
            all_layer_outputs.append((layer.name, layer_output.shape, layer_output))
        except Exception as e:
            # print(f"Error getting output of layer {layer.name}: {e}")
            pass
    # Print the shape of each layer's output
    # print(all_layer_outputs)
    for layer_name, output_shape, output_data in all_layer_outputs:
        print("#"*50)
        print(f"Image is being passed through: {layer_name}, Shape: {output_shape}")
        print("#"*50)
        print("Output data:", output_data)  # Print the output data

    print("Predicted Class:", predicted_class)
    
    print("Applying GRADCAM produce resultant Image....:")

    print("*"*50)
    print("Computing class activation map (CAM) using gradients...")
    print("*"*50)
    print("Computing the global average of the CAM...")
    print("*"*50)
    print("Multiplying the CAM with the gradients...")
    print("*"*50)
    print("Reducing the result to a single channel heatmap...")
    print("*"*50)
    print("Normalizing the heatmap...")
    print("*"*50)
    print("Process completed successfully.")
    print("*"*50)

    print("Heatmap Image Path:", heatmap_image_path)

    # Reset stdout

    sys.stdout = old_stdout

    output = mystdout.getvalue()

    return predicted_class, heatmap_image_path, additional_info_for_class, output



@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error="No image selected")

        file = request.files['image']

        if file.filename == '':
            return render_template('index.html', error="No image selected")

        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            predicted_class, heatmap_image_path, additional_info, output = predict(file_path)
            return render_template('index.html', result=predicted_class, image=file_path, heatmap=heatmap_image_path, additional_info=additional_info, output=output)

    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/performance')
def performance():
    return render_template('performance.html')

if __name__ == '__main__':
    app.run(debug=True)
