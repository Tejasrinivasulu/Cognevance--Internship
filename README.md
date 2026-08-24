<div align="center">

<h3>Cognevance Internship</h3>

<h1>Internship Task Portfolio</h1>

<p>
Three completed projects covering <b>Machine Learning</b>, <b>Deep Learning</b>, and <b>API Deployment</b>.
</p>

<p>
📊 Churn Prediction • 👕 Image Classification • 🚀 Flask REST API
</p>

<img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python">
<img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn">
<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch">
<img src="https://img.shields.io/badge/Flask-REST%20API-black?style=for-the-badge&logo=flask">
<img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge">

</div>

<hr>

<h2>✨ Project Overview</h2>

<p>
This repository contains all three Cognevance internship tasks. Task 1 trains and compares
classification models on telecom customer churn. Task 2 builds a Fashion-MNIST image
classifier with transfer learning. Task 3 deploys the Task 1 model as a Flask REST API
with a web interface and Docker support.
</p>

<p>
Each task folder has its own detailed README with full setup steps.
</p>

<h2>🛠️ Technologies</h2>

<p>
Python • Pandas • NumPy • Scikit-learn • XGBoost • Matplotlib • Seaborn •
PyTorch • Torchvision • Flask • Joblib • Docker • Jupyter Notebook
</p>

<hr>

<h2>📌 Tasks</h2>

<h3>Task 1 — Customer Churn Prediction</h3>

<p>
Predict whether a telecom customer is likely to churn using multiple Machine Learning
classification algorithms on the <b>Telco Customer Churn</b> dataset.
</p>

<ul>
<li>Data preprocessing and EDA</li>
<li>Logistic Regression, Random Forest, Decision Tree, SVM, KNN, XGBoost</li>
<li>Best model: <b>Logistic Regression</b> (ROC-AUC <b>0.8353</b>, Recall <b>0.7941</b>)</li>
</ul>

<p>Full details: <a href="Task1/README.md">Task1/README.md</a></p>

<h3>Task 2 — Fashion-MNIST Image Classifier</h3>

<p>
Classify clothing images using a pretrained <b>ResNet-18</b> network with
<b>PyTorch</b>, transfer learning, and data augmentation.
</p>

<ul>
<li>10 clothing categories (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot)</li>
<li>Adam optimizer and CrossEntropy loss</li>
<li>Test accuracy <b>92.9%</b> • Validation accuracy <b>93.0%</b> • Macro F1 <b>92.9%</b></li>
</ul>

<p>Full details: <a href="task2/README.md">task2/README.md</a></p>

<h3>Task 3 — Customer Churn Prediction API</h3>

<p>
Deploy the Task 1 <b>Logistic Regression</b> model as a <b>Flask REST API</b>.
Users can enter customer information through a web interface or REST API to
predict whether a customer is likely to <b>Churn</b> or <b>Stay</b>.
</p>

<ul>
<li>Real-time prediction</li>
<li>Web UI + REST API</li>
<li>Docker support</li>
</ul>

<p>Full details: <a href="task3/README.md">task3/README.md</a></p>

<hr>

<h2>📂 Project Structure</h2>

<pre>
cognevance-Internship/
│
├── Task1/          Customer Churn Prediction (ML)
│   ├── dataset/
│   ├── notebooks/
│   ├── src/
│   ├── model/
│   ├── images/
│   ├── run.bat
│   ├── predict.bat
│   ├── requirements.txt
│   └── README.md
│
├── task2/          Fashion-MNIST Image Classifier (DL)
│   ├── data/
│   ├── outputs/
│   ├── train.py
│   ├── inference.py
│   ├── train_image_classifier.ipynb
│   ├── setup.bat
│   ├── train.bat
│   ├── run.bat
│   ├── requirements.txt
│   └── README.md
│
├── task3/          Customer Churn Prediction API
│   ├── app/
│   ├── model/
│   ├── examples/
│   ├── screenshots/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── run.bat
│   ├── wsgi.py
│   ├── requirements.txt
│   └── README.md
│
└── README.md
</pre>

<hr>

<h2>▶️ How to Run</h2>

<h3>Task 1 — Customer Churn Prediction</h3>

<pre>
cd Task1
pip install -r requirements.txt
python src/run_pipeline.py
</pre>

<p><b>Or use the Windows batch file:</b></p>

<pre>
cd Task1
run.bat
</pre>

<h3>Task 2 — Fashion-MNIST Image Classifier</h3>

<pre>
cd task2
pip install -r requirements.txt
python train.py
python inference.py --image outputs/sample_sneaker.png --model outputs/best_model.pt
</pre>

<p><b>Or use the Windows batch files:</b></p>

<pre>
cd task2
setup.bat
train.bat
run.bat
</pre>

<h3>Task 3 — Customer Churn Prediction API</h3>

<pre>
cd task3
run.bat
</pre>

<p>
This creates <code>.venv</code>, installs packages, starts the app, and opens the browser.
</p>

<p>Then open:</p>

<pre>
http://127.0.0.1:8000
</pre>

<p>API documentation:</p>

<pre>
http://127.0.0.1:8000/api
</pre>

<p><b>Docker:</b></p>

<pre>
cd task3
docker compose up --build
</pre>

<p>If <code>pip</code> / <code>python</code> is not recognized:</p>

<pre>
python -m pip install -r requirements.txt
</pre>

<hr>

<div align="center">

<p>See each task README for notebooks, endpoints, and extra run options.</p>

</div>
