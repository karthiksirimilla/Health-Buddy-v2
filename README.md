# Health Buddy: AI-Powered Multilingual Healthcare Chatbot

**Empowering communities with accessible, AI-driven preliminary healthcare advice in multiple languages.** 🏥🤖🌍

---

## 🚀 Overview

**Health Buddy** is a smart, multilingual healthcare chatbot designed to offer instant, reliable medical guidance to users—especially in rural or underserved regions. By integrating AI for symptom-based diagnosis, voice analysis, and skin condition detection, it bridges the healthcare access gap and promotes early intervention.

> “Access to timely, trustworthy healthcare information should not be a privilege—it should be a basic right.”

---

## 🌟 Key Features

- 🌐 **Multilingual Support** – Breaks language barriers for better understanding (Currently available in English, हिंदी, తెలుగు, தமிழ், বাংলা)
- 🩺 **Symptom-based Diagnosis** – Analyzes user input to suggest possible conditions  
- 💊 **Medication & Remedies** – Recommends safe over-the-counter solutions  
- 🚨 **Emergency Alerts** – Flags critical conditions & suggests nearby hospitals  
- 🤖 **AI-Powered Skin Disease Detection** – Integrated via a Telegram bot  
- 🎙️ **Voice-based Cold Detection** – Analyzes user voice inputs for flu symptoms  

---

## 🧠 Tech Stack

| Domain        | Technologies Used                              |
|--------------|--------------------------------------------------|
| Frontend     | HTML5, CSS3, Bootstrap                          |
| Backend      | Python, Flask                                   |
| AI/ML        | PyTorch, OpenCV, YOLO V8                        |
| Bot Framework| Telegram Bot API                                |
| Others       | NLTK, Google Translate API, Botpress            |

---

## 🧪 How to Run

Follow the steps below to set up and run the Health Buddy project on your local machine:

---

### 1️⃣ **Clone the Repository**

```bash
git clone [https://github.com/youre/health-buddy.git](https://github.com/karthiksirimilla/Health-Buddy.git)
cd Health-Buddy
```
### 2️⃣ **Create a Virtual Environment**

```bash
python -m venv .venv
```
  * Activate it:
  
    * On **Windows**:
    
        ```bash
        .venv\Scripts\activate
        ```
    * On **macOS/Linux**:
    
        ```bash
        source .venv/bin/activate
        ```
    
### 3️⃣ **Install Dependencies**

  ```bash
  pip install -r requirements.txt
  ```

### 4️⃣ **Run the Application**
  ```bash
  python app.py
  ```

The Flask server will start running on http://127.0.0.1:5000.
Open this in your browser to interact with Health Buddy. 🌐

## 🎯 **Future Scope**

- Tongue-based illness detection using image processing

- Voice assistant integration for hands-free interaction

- Thermal camera-based fever detection

- Community health analytics and predictive insights

- More disease specializations

