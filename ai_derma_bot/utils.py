from ultralytics import YOLO
from PIL import Image
import os
import cv2
import numpy as np
from datetime import datetime
from gtts import gTTS

model_path = os.path.join("ai_derma_bot", "derma_bot.pt")
model = YOLO(model_path)

# Professional prescriptions for each condition
prescription_map = {
    "Chickenpox": "Maintain proper hygiene. Avoid scratching lesions. Use calamine lotion and antihistamines for itching. Consult your physician for antiviral therapy if needed.",
    "Eczema": "Apply prescribed emollients regularly. Avoid known triggers. In case of flare-ups, use topical corticosteroids as advised by a dermatologist.",
    "Eruptive-Xanthoma": "This may indicate underlying lipid metabolism disorders. Seek immediate medical advice for systemic evaluation and lipid-lowering therapy.",
    "Leukocytoclastic-Vasculitis": "Use anti-inflammatory medications and rest. Immediate consultation with a healthcare provider is recommended for further assessment.",
    "Monkeypox": "Isolate the patient to prevent transmission. Supportive care is essential. Seek medical evaluation for antiviral treatment if symptoms persist.",
    "Ringworm": "Apply antifungal creams such as clotrimazole or terbinafine. Keep affected areas clean and dry. Medical follow-up is recommended.",
    "Spider-Angioma": "Typically harmless. However, if multiple lesions are present, a liver function evaluation is advised. Cosmetic treatments include laser therapy.",
    "Xanthelasma": "Associated with cholesterol imbalance. Consider lipid profile testing. Surgical removal may be performed for cosmetic reasons.",
    "herpes-zoster": "Administer antiviral drugs like acyclovir within 72 hours of symptom onset. Manage pain with analgesics and monitor for complications.",
    "vitiligo": "Treatment may include corticosteroids, calcineurin inhibitors, or phototherapy. Consultation with a dermatologist is necessary for personalized management."
}

def detect_and_respond(image_path):
    image = Image.open(image_path).convert("RGB")
    results = model.predict(image, conf=0.25)
    names = model.names

    if not results or not results[0].boxes:
        return {
            "label": "None",
            "confidence": "0%",
            "instances": 0,
            "prescription": "No visible skin issue detected. However, a clinical consultation is advised for confirmation.",
            "image_url": "/static/results/default.jpg",
            "audio_url": ""
        }

    boxes = results[0].boxes
    classes = boxes.cls.tolist()
    confs = boxes.conf.tolist()
    image_np = np.array(image)

    for box, cls, conf in zip(boxes.xyxy.tolist(), classes, confs):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[int(cls)]} ({int(conf * 100)}%)"
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    result_dir = "static/results"
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    annotated_filename = f"annotated_{timestamp}.jpg"
    result_path = os.path.join(result_dir, annotated_filename)
    cv2.imwrite(result_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    # Identify most frequent label
    counts = {}
    for cls in classes:
        name = names[int(cls)]
        counts[name] = counts.get(name, 0) + 1
    top_label = max(counts.items(), key=lambda x: x[1])[0]
    top_conf = int(max(confs) * 100)
    prescription = prescription_map.get(
        top_label,
        "For this condition, please consult a certified dermatologist for a personalized treatment plan."
    )

    # Generate speech from prescription
    speech_text = f"Detected {top_label}. Confidence is {top_conf} percent. Suggested prescription: {prescription}"
    audio_filename = f"recommend_{timestamp}.mp3"
    audio_path = os.path.join(result_dir, audio_filename)
    tts = gTTS(speech_text)
    tts.save(audio_path)

    return {
        "label": top_label.capitalize(),
        "confidence": f"{top_conf}%",
        "instances": counts[top_label],
        "prescription": prescription,
        "image_url": f"/static/results/{annotated_filename}",
        "audio_url": f"/static/results/{audio_filename}"
    }
