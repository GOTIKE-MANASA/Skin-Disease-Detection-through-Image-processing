from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
import os
import base64
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Load the model
try:
    model_path = "skin_disease_model.keras"
    model = load_model(model_path)
    print("Model loaded successfully from:", model_path)
except Exception as e:
    print(f"Error loading model: {e}")

app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define classes based on training data
classes = [
    'Atopic Dermatitis', 'Basal Cell Carcinoma (BCC)',
    'Benign Keratosis-like Lesions (BKL)', 'dry', 'Eczema',
    'Melanocytic Nevi (NV)', 'Melanoma', 'normal', 'oily',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Seborrheic Keratoses and other Benign Tumors',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Warts Molluscum and other Viral Infections'
]
skin_type=['dry','normal','oily']

disease_info = {
'Atopic Dermatitis': {
        'symptoms': 'Itching (often severe, especially at night),Dry,scaly skin,Red patches, particularly on the face, neck, and limbs,Thickened skin from prolonged scratching,Fluid-filled blisters that may ooze and crust over,Darkened or swollen skin in affected areas',
        'causes': 'Genetics: Family history of eczema, allergies, or asthma,Immune response: Overactive immune system reaction to irritants,Environmental triggers: Such as dust, allergens, soaps, or changes in weather,Skin barrier issues: Reduced ability to retain moisture, leading to dry skin.',
        'transmission': ' not contagious; it cannot be spread from person to person.',
        'treatment': 'Topical corticosteroids: To reduce inflammation and itching,Moisturizers: To keep skin hydrated and prevent dryness,Oral medications: Antihistamines for itch relief; sometimes antibiotics if infection occurs,Light therapy (phototherapy): For severe cases, under medical supervision.',
        'prevention': 'Moisturize daily: To keep skin hydrated,Avoid known triggers: Such as soaps, certain fabrics, and extreme temperatures,Use gentle skin products: Prefer fragrance-free and hypoallergenic products,Wear loose clothing: To avoid irritation,Maintain a cool, humid environment: To prevent skin dryness.'
    },
    'Basal Cell Carcinoma (BCC)': {
        'symptoms': 'Small, shiny or pearly bumps on the skin,Flat, scaly patches resembling scars,Pink or red growths with raised edges,Bleeding or oozing sore that does not heal',
        'causes': 'UV Radiation- Prolonged exposure to ultraviolet (UV) rays from the sun or tanning beds,Genetic Factors-Certain genetic conditions, such as Gorlin syndrome, can increase the risk,Fair Skin- People with lighter skin tones are more prone to BCC',
        'transmission': 'BCC is not contagious. It cannot be transmitted from person to person.',
        'treatment': 'Surgical Excision: Complete removal of the tumor and a margin of healthy skin,Curettage and Electrodesiccation: Scraping away the tumor and using heat to destroy any remaining cells,Cryotherapy: Freezing the tumor with liquid nitrogen,Topical Medications: Prescription creams or ointments for superficial BCC,Radiation Therapy: Targeted radiation for BCC in hard-to-treat areas',
        'prevention': 'Sun Protection: Use sunscreen with SPF 30 or higher, wear protective clothing, and seek shade,Avoid Tanning Beds: Tanning beds emit harmful UV rays,Regular Skin Checks: Conduct self-examinations and see a dermatologist annually,Early Detection: Promptly consult a doctor about new or changing skin lesions'
    },
    'Benign Keratosis-like Lesions (BKL)': {
        'symptoms': 'Small, rough, scaly patches on the skin,Lesions can be light or dark brown, gray, or even black,Usually painless, though may itch or cause discomfort if irritated,Commonly appear on sun-exposed areas such as the face, hands, shoulders, and arms',
        'causes': 'Mostly due to aging and sun exposure,Genetic factors may play a role, especially in those with a family history of similar skin conditions,Not caused by any infectious agent or virus',
        'transmission': 'BKL is not contagious and cannot be spread from person to person.',
        'treatment': 'Treatment is generally not required unless lesions are causing discomfort or cosmetic concern,Common removal methods include cryotherapy (freezing), curettage, or laser therapy,Topical treatments may help in mild cases, though they are less common',
        'prevention': 'Limiting sun exposure and using sunscreen to reduce the risk of developing new lesions,Wearing protective clothing in the sun,Regular skin check-ups for early detection of any suspicious growths'
    },
    'Eczema': {
        'symptoms': 'Itching and Redness: Commonly on hands, feet, face, or inside the elbows and knees,Dry, Scaly Patches: Skin may become thickened and rough,Blisters and Crusting: In some cases, small blisters can form, especially if the skin becomes infected.',
        'causes': 'Genetics: Family history of eczema, allergies, or asthma increases risk,Immune System: An overactive immune response leads to inflammation,Environmental Factors: Exposure to allergens, stress, or harsh soaps can trigger eczema.',
        'transmission': 'Non-Contagious: Eczema cannot be spread from person to person.',
        'treatment': 'Moisturizers: Daily application to keep the skin hydrated,Topical Steroids: Used to reduce inflammation and itching,Antibiotics: For infections if the skin is broken,Immunomodulators: For severe cases, medications like cyclosporine or dupilumab may be used.',
        'prevention': 'Avoid Triggers: Identify and avoid things that cause flare-ups, like allergens or irritating soaps,Moisturize Regularly: Apply moisturizers daily to prevent dryness,Gentle Skin Care: Use mild, unscented soaps and lukewarm water for bathing,Wear Soft Fabrics: Avoid rough or scratchy materials like wool.'
    },
    'Melanocytic Nevi (NV)': {
        'symptoms': 'Small, round, or oval brown/black spots, also known as moles,May be flat or raised, with a smooth or rough texture,Often appear in childhood or adolescence.',
        'causes': 'Caused by clusters of melanocytes (pigment-producing cells) in the skin,Genetic factors and sun exposure can influence the development',
        'transmission': 'Not contagious; often genetically influenced.',
        'treatment': 'Often no treatment is needed,Removal may be recommended if there is a risk of melanoma (cancer) or cosmetic reasons.',
        'prevention': 'Limit sun exposure and use sunscreen,Regularly monitor moles for changes in size, shape, or color.'
    },
    'Melanoma': {
        'symptoms': 'New or changing mole, often dark in color,Asymmetry, irregular borders, color variations, and diameter larger than 6mm (ABCDE rule),Itching, tenderness, or bleeding from a mole.',
        'causes': 'Caused by genetic mutations in melanocytes, often triggered by UV radiation,Family history and fair skin increase the risk.',
        'transmission': 'Not contagious',
        'treatment': 'Surgery to remove the melanoma,Advanced cases may require immunotherapy, chemotherapy, or radiation therapy',
        'prevention': 'Regular self-exams for early detection,Avoid excessive sun exposure and use sunscreen,Protective clothing and regular dermatologist visits.'
    },
    'Psoriasis pictures Lichen Planus and related diseases': {
        'symptoms': 'Red, scaly patches on the skin, often on elbows, knees, and scalp,Itching, burning, or soreness in affected areas,Thickened or pitted nails.',
        'causes': 'An autoimmune disorder where the immune system attacks skin cells,Triggered by stress, infections, injuries, and certain medications.',
        'transmission': 'Not contagious',
        'treatment': 'Topical treatments (steroids, vitamin D analogs),Phototherapy and systemic medications for severe cases,Biologic drugs targeting immune responses',
        'prevention': 'Manage stress and avoid known triggers,Moisturize skin regularly,Maintain a healthy diet and avoid smoking'
    },
    'Seborrheic Keratoses and other Benign Tumors': {
        'symptoms': 'Brown, black, or light tan growths, often with a waxy or "stuck-on" appearance,Commonly found on the face, chest, shoulders, and back',
        'causes': 'Non-cancerous growths that appear with age,Genetic factors may play a role',
        'transmission': 'Not contagious',
        'treatment': 'Often no treatment is necessary,Removal for cosmetic reasons or if irritated, using cryotherapy, laser, or surgery',
        'prevention': 'No specific prevention methods, but regular monitoring can help distinguish from more serious conditions'
    },
    'Tinea Ringworm Candidiasis and other Fungal Infections': {
        'symptoms': 'Tinea: Red, circular rash with raised edges and clear center, often on the scalp, body, or feet (athlete’s foot),Candidiasis: Red, itchy rash, often in moist areas like under breasts, between fingers, or groin area',
        'causes': 'Fungal infections caused by dermatophytes (ringworm) or Candida species (candidiasis),Warm, moist environments promote fungal growth.',
        'transmission': 'Contagious through direct contact or sharing personal items.',
        'treatment': 'Topical or oral antifungal medications,Keeping skin dry and clean',
        'prevention': 'Avoid sharing personal items,Keep skin dry, especially in skin folds,Wear breathable clothing and shoes'
    },
    'Warts Molluscum and other Viral Infections': {
        'symptoms': 'Warts: Small, rough, skin-colored growths, commonly on hands and feet,Molluscum contagiosum: Small, painless, pearl-like bumps, often with a dimple in the center.',
        'causes': 'Warts are caused by human papillomavirus (HPV),Molluscum is caused by the molluscum contagiosum virus',
        'transmission': 'Contagious through direct contact or sharing personal items',
        'treatment': 'Warts: Cryotherapy, salicylic acid, or laser removal,Molluscum: Often resolves on its own, but cryotherapy or topical treatments may help',
        'prevention': 'Avoid direct contact with infected individuals,Do not share personal items,Practice good hygiene, especially in public places like pools and gyms.'
    }
}

# Global variable to hold the prediction result
prediction_result = ""
symptoms=""
causes=""
transmission=""
treatment=""
prevention=""


# Prediction function
def predict_image(image_path):
    try:
        print(f"Starting prediction for image: {image_path}")

        # Load and preprocess the image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions
        prediction = model.predict(img_array)
        print(f"Raw prediction output: {prediction}")

        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx]
        types=[prediction[0][3],prediction[0][7],prediction[0][8]]
        type_idx=np.argmax(types)
        print(f"Predicted class: {classes[class_idx]}, Confidence: {confidence}")
        predicted_label=classes[class_idx]


        # Determine result
        if confidence < 0.7:  # Confidence threshold
            #print(f"Skin Type Detected: {classes[class_idx]}")
            return skin_type[type_idx]
        else:
            #print(f"Skin Disease Detected: {classes[class_idx]}")
            return classes[class_idx]

    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Error during prediction."

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result')
def result_page():
    global prediction_result
    global symptoms
    global causes
    global transmission
    global treatment
    global prevention

    if prediction_result in ['oily','normal','dry']:
        return render_template('result.html', result=prediction_result)
    else:
        return render_template('result1.html', label=prediction_result,
                               symptoms=symptoms,
                               causes=causes, transmission=transmission,
                               treatment=treatment, prevention=prevention)

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join('static/uploads', filename)

        # Ensure the uploads directory exists
        os.makedirs('uploads', exist_ok=True)


        # Save the file
        file.save(file_path)
        #print(f"File uploaded successfully: {file_path}")

        # Confirm the file is saved correctly
        if not os.path.exists(file_path):
            print(f"Error: The file was not saved at {file_path}")
        else:
            print(f"File successfully saved at {file_path}")

        # Predict the image
        global prediction_result
        global symptoms
        global causes
        global transmission
        global treatment
        global prevention
        global disease_info
        prediction_result = predict_image(file_path)


        # Get the disease information
        if prediction_result in disease_info:
            #disease_label = disease_info[prediction_result]
            symptoms = disease_info[prediction_result]['symptoms']
            causes = disease_info[prediction_result]['causes']
            transmission = disease_info[prediction_result]['transmission']
            treatment = disease_info[prediction_result]['treatment']
            prevention = disease_info[prediction_result]['prevention']
        else:
            symptoms = "Unknown"
            causes = "Unknown"
            transmission = "Unknown"
            treatment = "Unknown"
            prevention = "Unknown"

        # Clean up
        #os.remove(file_path)
        return redirect(url_for('result_page'))
    except Exception as e:
        print(f"Error in upload route: {e}")
        return redirect(url_for('result_page'))

@app.route('/camera', methods=['POST'])
def camera_capture():
    try:
        data = request.json['image']
        file_path = "camera_image.jpg"

        # Decode the base64 image and save it
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(data.split(',')[1]))  # Remove the base64 header
        print(f"Image captured and saved as: {file_path}")

        # Predict the image
        global symptoms
        global causes
        global transmission
        global treatment
        global prevention
        global disease_info
        global prediction_result
        prediction_result = predict_image(file_path)
        print(f"hey I am {prediction_result}")

        if prediction_result in disease_info:
            #disease_label = disease_info[prediction_result]
            symptoms = disease_info[prediction_result]['symptoms']
            causes = disease_info[prediction_result]['causes']
            transmission = disease_info[prediction_result]['transmission']
            treatment = disease_info[prediction_result]['treatment']
            prevention = disease_info[prediction_result]['prevention']
        else:
            symptoms = "Unknown"
            causes = "Unknown"
            transmission = "Unknown"
            treatment = "Unknown"
            prevention = "Unknown"

        # Clean up
        os.remove(file_path)
        return redirect(url_for('result_page'))
    except Exception as e:
        print(f"Error in camera route: {e}")
        prediction_result = "Error during prediction."
        return redirect(url_for('result_page'))

if __name__ == '__main__':
    app.run(debug=True)
