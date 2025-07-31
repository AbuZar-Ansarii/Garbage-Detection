import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Class labels
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Load trained model
model = load_model("garbage_classifier_cnn.h5")

# IP Webcam URL
url = 'http://192.168.38.85:8080/video'
cap = cv2.VideoCapture(url)

# Loop through frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and apply threshold for motion/object detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Find contours (possible garbage regions)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through contours
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter small noise
        if w > 80 and h > 80:
            roi = frame[y:y+h, x:x+w]

            # Preprocess ROI for prediction
            try:
                roi_resized = cv2.resize(roi, (150, 150))
                img_array = img_to_array(roi_resized)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                # Predict class
                prediction = model.predict(img_array, verbose=0)
                pred_class = class_labels[np.argmax(prediction)]
                confidence = np.max(prediction)

                # Draw rectangle and label
                box_color = (0, 255, 0)
                label = f"{pred_class} ({confidence*100:.1f}%)"
                cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

            except Exception as e:
                print("Error in ROI:", e)

    # Show live video
    cv2.imshow("Garbage Detection Live Feed", frame)

    # Break on 'q'
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
