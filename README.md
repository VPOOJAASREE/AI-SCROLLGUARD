# AI-SCROLLGUARD


---

# 📱 AI ScrollGuard – Smart Digital De-Addiction System

AI ScrollGuard is a **web-based digital wellness and productivity monitoring system** designed to help users **track screen time, detect overuse, predict future behavior, and reduce digital addiction** using AI-driven logic.

This project is developed as a **CSE Mini Project** and demonstrates the application of **AI, data analytics, and web technologies** for social good.

---

## 🎯 Key Features

* ✅ Daily screen-time and task entry
* 🟢🟥 AI-based behavior classification (Healthy / Overuse)
* 🔔 Real-time alerts and task reminders
* 🔥 Streak tracking and productivity scoring
* 📊 Insights dashboard with usage trends
* 🤖 AI-based addiction risk prediction
* 🔮 Tomorrow usage prediction
* 🧪 What-if simulation for future behavior analysis
* 🌙 Light/Dark mode UI

---

## 🏗️ Project Architecture

**Frontend**

* HTML, CSS, JavaScript
* Chart.js for data visualization

**Backend**

* Python Flask (REST API)
* Flask-CORS for frontend communication

**AI / Logic**

* Rule-based AI for classification & prediction
* Machine Learning (Random Forest) for addiction risk prediction

**Storage**

* In-memory + CSV-based data storage (prototype level)

---

## ⚙️ Technologies Used

* **Frontend:** HTML, CSS, JavaScript
* **Backend:** Python, Flask
* **AI / ML:** NumPy, Scikit-learn, Joblib
* **Visualization:** Chart.js
* **Tools:** VS Code, Browser

---

## 📁 Project Structure

```
AI-ScrollGuard/
│
├── backend/
│   ├── app.py
│   ├── risk_model.pkl
│   └── risk_dataset.csv
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── README.md
└── requirements.txt
```

---

## 🛠️ Installation & Setup

### 1️⃣ Prerequisites

* Python 3.9 or above
* Web browser (Chrome / Edge)

---

### 2️⃣ Install Required Libraries

```bash
pip install flask flask-cors numpy pandas scikit-learn joblib
```

---

### 3️⃣ Run the Application

```bash
python app.py
```

---

### 4️⃣ Open in Browser

```
http://127.0.0.1:5000
```

---

## 🧪 How to Use

1. Enter daily screen usage (in minutes)
2. Enter today’s important task
3. Submit data to receive:

   * Usage classification
   * Alerts and reminders
   * Productivity streak update
4. View:

   * Dashboard summary
   * Usage trends (Insights tab)
   * AI addiction risk (Risk Analysis tab)
5. Use **Load Sample Data** for demo purposes

---

## 📊 AI Prediction Logic

* **Classification:** Threshold-based (≤ 60 mins → Green, > 60 mins → Red)
* **Tomorrow Prediction:** Weighted average of past usage
* **Risk Prediction:** Random Forest ML model
* **What-if Simulation:** Predicts risk for planned future usage

---

## 🌍 Social Impact

* Encourages healthy digital habits
* Improves productivity and focus
* Helps prevent digital addiction
* Suitable for students, parents, educators, and wellness centers

---

## 🚀 Future Enhancements

* Automatic mobile app usage tracking
* User authentication & profiles
* Cloud database integration
* Mobile app version
* Advanced deep learning models
* Parental control dashboard

---

## ⚠️ Limitations

* Manual data entry (simulation-based)
* No live mobile data access
* Prototype-level implementation

---

## 📚 Academic Information

* **Project Type:** Mini Project (CSE)
* **Application Domain:** AI for Digital Well-being
* **Institution:** Saveetha Engineering College
* **Purpose:** Academic & learning use

---

## 👩‍💻 Developer

**V. Poojaa Sree**
Department of Computer Science and Engineering

---

## 📜 License

This project is intended for **educational purposes only**.

---




