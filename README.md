# 🧪 Neykuri Pattern Classifier

A **Streamlit web app** for classifying Siddha medical constitution patterns from **Neikuri (oil drop) images** using a deep learning model.

> 🔬 Developed as a Capstone project (2024–25) to bridge traditional Indian diagnostic methods with modern AI.

---

## 📸 Features

- Upload up to **5 Neikuri images**
- Automatically crops and focuses on the oil drop region
- Predicts one of the following Siddha constitution types:
  - **Kabam**
  - **Pithakabam**
  - **Pithalipitham**
  - **Pitham**
  - **Pithavatham**
- Provides a short medical explanation for each prediction

---

## 🚀 Run Locally

### 🧰 Requirements

Install dependencies:
pip install -r requirements.txt

##▶️ Run the app
streamlit run app.py

🔄 Model Not Included
⚠️ Note: The trained model file (densenet121.h5) is not included in this repository due to GitHub's file size limit.

To use the app:

Upload the model manually to your local folder
OR
Modify app.py to download the model from Google Drive or other storage

Want help with that? Open an issue or contact the author below.

📂 Project Structure
File	Description
app.py	Streamlit app source code
requirements.txt	Python dependencies
README.md	This file
(You must add the .h5 model file yourself)	

👩‍🔬 Author
Hrishita Chakraborty
Capstone Project, 2024–25
📧 hrishitachakraborty.2022@gmail.com

📜 License
For academic and research use only. Please contact the author for reuse permissions or extensions.

