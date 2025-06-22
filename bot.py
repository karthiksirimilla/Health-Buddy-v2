import os
import time
import tempfile
from gtts import gTTS
from ultralytics import YOLO
import telebot
import glob
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables


# Load Telegram Bot Token from Environment Variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Load YOLO Model (Ensure `best.pt` is inside the repo)
model_path = "best.pt" 
yolo_model = YOLO(model_path)

# Disease Prescriptions
disease_prescriptions = {
    "Chickenpox": "Use calamine lotion and avoid scratching.",
    "Eczema": "Apply fragrance-free moisturizers and avoid triggers.",
    "Ringworm": "Use topical antifungal creams like clotrimazole.",
    "herpes-zoster": "Take antiviral medications like acyclovir.",
    "vitiligo": "Consult a dermatologist for treatment options."
}

def detect_and_save(input_path, conf=0.25):
    """Run YOLO detection on the input file and save output images."""
    results = yolo_model.predict(source=input_path, conf=conf, save=True)
    
    latest_folder = max(glob.glob('/content/runs/detect/*'), key=os.path.getmtime)
    output_images = glob.glob(f'{latest_folder}/*.jpg')

    detected_diseases = set()
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = yolo_model.names[class_id]
            detected_diseases.add(class_name)

    return output_images, detected_diseases

def generate_audio_report(diseases):
    """Generate an audio prescription report."""
    if not diseases:
        text = "No diseases detected."
    else:
        text = "Detected diseases and treatments:\n"
        for disease in diseases:
            text += f"{disease}: {disease_prescriptions.get(disease, 'Consult a doctor.')}\n"

    tts = gTTS(text)
    temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(temp_audio_path)
    return temp_audio_path

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Welcome to **Health Buddy**! Send an image to detect skin diseases and get treatment suggestions.")

@bot.message_handler(content_types=['photo'])
def handle_image(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(downloaded_file)
            temp_image_path = temp_file.name

        output_images, detected_diseases = detect_and_save(temp_image_path)

        if detected_diseases:
            response = "Detected Diseases:\n" + "\n".join(f"- {d}" for d in detected_diseases)
            bot.reply_to(message, response)

            for disease in detected_diseases:
                bot.reply_to(message, f"Treatment for {disease}: {disease_prescriptions.get(disease, 'Consult a doctor.')}")

            audio_path = generate_audio_report(detected_diseases)
            if audio_path:
                with open(audio_path, 'rb') as audio_file:
                    bot.send_audio(message.chat.id, audio_file)
                os.remove(audio_path)

        else:
            bot.reply_to(message, "No diseases detected.")

        for img_path in output_images:
            with open(img_path, 'rb') as img_file:
                bot.send_photo(message.chat.id, img_file)
            os.remove(img_path)

        os.remove(temp_image_path)

    except Exception as e:
        bot.reply_to(message, "Error processing the image.")
        print(f"Error: {e}")

# Keep the bot running continuously
while True:
    try:
        bot.polling(none_stop=True)
    except Exception as e:
        print(f"Bot crashed! Restarting in 5 seconds... Error: {e}")
        time.sleep(5)
