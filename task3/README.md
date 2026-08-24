<div align="center">

<h3>Cognevance Internship - Task 3</h3>

<h1>🚀 Customer Churn Prediction API</h1>

<p>
Deploy a <b>Machine Learning</b> model using <b>Flask REST API</b> to predict customer churn in real time.
</p>

<p>
🤖 Machine Learning • ⚡ Flask API • 🐳 Docker • 📊 Real-Time Prediction
</p>

<img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python">
<img src="https://img.shields.io/badge/Flask-REST%20API-black?style=for-the-badge&logo=flask">
<img src="https://img.shields.io/badge/Docker-Container-blue?style=for-the-badge&logo=docker">
<img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge">

</div>

<hr>

<h2>✨ Project Overview</h2>

<p>
This project deploys the <b>Customer Churn Prediction</b> Machine Learning model as a
<b>Flask REST API</b>. Users can enter customer information through a web interface
or REST API to predict whether a customer is likely to <b>Churn</b> or <b>Stay</b>.
</p>

<h2>🛠️ Technologies</h2>

<p>
Python • Flask • Scikit-learn • Pandas • NumPy • Joblib • HTML • CSS • Docker
</p>

<h2>🎯 Model</h2>

<p>
The deployed model is the <b>Logistic Regression</b> classifier trained in
Task 1 using the <b>Telco Customer Churn</b> dataset.
</p>

<ul>
<li>Logistic Regression</li>
<li>Data Preprocessing Pipeline</li>
<li>Real-Time Prediction</li>
<li>REST API Deployment</li>
<li>Docker Support</li>
</ul>

<h2>🏆 Model Performance</h2>

<p align="center"><b>Table 2.4 – Customer Churn Model Performance</b></p>
<p align="center">
<img src="screenshots/table-2.4-model-performance.png" width="700">
</p>

<hr>

<h2>📂 Project Structure</h2>

<pre>
Customer-Churn-Prediction-API/
│
├── app/
│   ├── main.py
│   ├── model.py
│   ├── templates/
│   └── static/
│
├── model/
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   └── metrics.csv
│
├── examples/
│
├── screenshots/
│   ├── home.png
│   ├── predict.png
│   ├── result.png
│   └── docker.png
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── run.bat
├── wsgi.py
└── README.md
</pre>

<hr>

<h2>▶️ How to Run</h2>

<h3>1️⃣ Easiest (Windows)</h3>

<pre>
run.bat
</pre>

<p>
This creates <code>.venv</code>, installs packages, starts the app, and opens the browser.
</p>

<h3>2️⃣ Manual Install</h3>

<pre>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
</pre>

<p>If <code>pip</code> is not recognized:</p>

<pre>
python -m pip install -r requirements.txt
</pre>

<h3>3️⃣ Start the Application</h3>

<pre>
python -m app.main
</pre>

<h3>4️⃣ Open in Browser</h3>

<pre>
http://127.0.0.1:8000
</pre>

<h3>5️⃣ API Documentation</h3>

<pre>
http://127.0.0.1:8000/api
</pre>

<hr>

<h2>💻 Run on Another Laptop</h2>

<ol>
<li>Zip the project <b>without</b> <code>.venv</code></li>
<li>Unzip on the other laptop</li>
<li>Install Python 3</li>
<li>Run <code>run.bat</code></li>
</ol>

<pre>
.\run.bat
</pre>

<hr>

<h2>📡 API Endpoints</h2>

<p align="center"><b>Table 2.5 – Customer Churn Prediction API Endpoints</b></p>
<p align="center">
<img src="screenshots/table-2.5-api-endpoints.png" width="800">
</p>

<hr>

<h2>📸 Generated Outputs</h2>

<h3>Figure 2.8 – Customer Churn Prediction Home Page</h3>
<p align="center">
<img src="screenshots/home.png" width="700" alt="Figure 2.8 Home Page">
</p>

<h3>Figure 2.9 – Customer Prediction Page</h3>
<p align="center">
<img src="screenshots/predict.png" width="700" alt="Figure 2.9 Prediction Page">
</p>

<h3>Figure 2.10 – Customer Prediction Result</h3>
<p align="center">
<img src="screenshots/result.png" width="700" alt="Figure 2.10 Prediction Result">
</p>

<h3>Figure 2.11 – Docker Deployment</h3>
<p align="center">
<img src="screenshots/docker.png" width="700" alt="Figure 2.11 Docker Deployment">
</p>

<hr>

<h2>📊 Sample Prediction</h2>

<pre>
Customer Prediction

Prediction : No Churn

Probability : 9.0%

Status : Customer is likely to stay (Low Risk).
</pre>

<hr>

<h2>🐳 Docker</h2>

<pre>
docker compose up --build
</pre>

<p>Then open <code>http://127.0.0.1:8000</code></p>

<hr>

<div align="center">

⭐ <b>If you like this project, don't forget to Star the repository!</b>

</div>
