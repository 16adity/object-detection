import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

model = ResNet50(weights='imagenet')

def recognize_animal(frame):
    # Preprocess the input frame
    x = image.img_to_array(frame)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make predictions using the pre-trained ResNet-50 model
    predictions = model.predict(x)

    # Decode predictions to human-readable labels
    animal_predictions = decode_predictions(predictions, top=1)[0]
    print(animal_predictions)

    animal_classes = ['hen', 'iguana', 'snake', 'spider', 'tortoise', 'turtle', 'bird', 'porcupine', 'gecko', 'egg']

    # Check if the predicted class is an animal
    if animal_predictions[0][1] in animal_classes:
        return animal_predictions[0][1]
    else:
        return 'none'

# Capture video from the webcam
def detection_using_camera():
    cap = cv2.VideoCapture(0)

    while True:
        # Read the frame from the webcam
        ret, frame = cap.read()

        if not ret:
            break

        # Recognize the animal in the frame
        frame1 = cv2.resize(frame, (224, 224))
        predicted_animal = recognize_animal(frame1)

        # Display the prediction on the frame
        text_to_display = 'Predicted Animal: ' + predicted_animal
        if predicted_animal == 'none':
            text_to_display = 'Detected: ' + predicted_animal
        cv2.putText(frame, text_to_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow('Animal Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

def detection_using_image():
    # Path to the image file
    img_path = r'C:/Users/asus/Desktop/GITAM/animal_detection/animals picture/iphone.png'

    # Example usage of recognize_animal function
    img = image.load_img(img_path, target_size=(224, 224))
    predicted_animal = recognize_animal(img)
    

detection_using_image()
