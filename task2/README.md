<div align="center">

<h3>Cognevance Internship - Task 2</h3>

<h1>👕 Fashion-MNIST Image Classifier</h1>

<p>
Classify clothing images using <b>Deep Learning</b> with <b>PyTorch</b> and <b>Transfer Learning</b>.
</p>

<p>
🧠 Transfer Learning • 🎨 Data Augmentation • 📈 Training Curves • 🤖 Image Classification
</p>

<img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python">
<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch">
<img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge">

</div>

<hr>

<h2>✨ Project Overview</h2>

<p>
This project implements an image classification model using a pretrained
<b>ResNet-18</b> network on the <b>Fashion-MNIST</b> dataset. The model uses
transfer learning, data augmentation, and evaluation metrics to classify
different clothing categories.
</p>

<h2>🛠️ Technologies</h2>

<p>
Python • PyTorch • Torchvision • NumPy • Matplotlib • Scikit-learn • Pillow • Jupyter Notebook
</p>

<h2>🎯 Dataset</h2>

<p>
<b>Fashion-MNIST</b> contains 70,000 grayscale images (28×28 pixels) across 10 clothing categories.
</p>

<ul>
<li>T-Shirt / Top</li>
<li>Trouser</li>
<li>Pullover</li>
<li>Dress</li>
<li>Coat</li>
<li>Sandal</li>
<li>Shirt</li>
<li>Sneaker</li>
<li>Bag</li>
<li>Ankle Boot</li>
</ul>

<h2>🧠 Model</h2>

<ul>
<li>Pretrained ResNet-18</li>
<li>Transfer Learning</li>
<li>Data Augmentation</li>
<li>Adam Optimizer</li>
<li>CrossEntropy Loss</li>
</ul>

<h2>🏆 Model Performance</h2>

<table>
<tr>
<th>Metric</th>
<th>Score</th>
</tr>

<tr>
<td><b>Test Accuracy</b></td>
<td><b>92.9%</b></td>
</tr>

<tr>
<td><b>Validation Accuracy</b></td>
<td><b>93.0%</b></td>
</tr>

<tr>
<td><b>Macro F1 Score</b></td>
<td><b>92.9%</b></td>
</tr>

</table>

<hr>

<h2>📂 Project Structure</h2>

<pre>
Fashion-MNIST-Image-Classifier/
│
├── data/
├── outputs/
│   ├── best_model.pt
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── metrics.json
│   └── sample_sneaker.png
│
├── train.py
├── inference.py
├── train_image_classifier.ipynb
├── setup.bat
├── train.bat
├── run.bat
├── requirements.txt
└── README.md
</pre>

<hr>

<h2>▶️ How to Run</h2>

<h3>1️⃣ Install Packages</h3>

<pre>
pip install -r requirements.txt
</pre>

<p>If <code>pip</code> is not recognized:</p>

<pre>
python -m pip install -r requirements.txt
</pre>

<h3>2️⃣ Train the Model</h3>

<pre>
python train.py
</pre>

<p><b>Or simply run:</b></p>

<pre>
train.bat
</pre>

<h3>3️⃣ Run Inference</h3>

<pre>
python inference.py --image outputs/sample_sneaker.png --model outputs/best_model.pt
</pre>

<p><b>Or simply run:</b></p>

<pre>
run.bat
</pre>

<h3>4️⃣ Open Notebook</h3>

<pre>
python -m notebook train_image_classifier.ipynb
</pre>

<p>
In <b>VS Code / Cursor</b>, open
<code>train_image_classifier.ipynb</code>,
select the Python kernel and click <b>Run All</b>.
</p>

<hr>

<h2>📸 Generated Outputs</h2>

<h3>📈 Training Curves</h3>

<p align="center">
<img src="outputs/training_curves.png" width="700">
</p>

<h3>🧩 Confusion Matrix</h3>

<p align="center">
<img src="outputs/confusion_matrix.png" width="650">
</p>

<h3>📊 Metrics</h3>

<p>
Training generates:
</p>

<ul>
<li>✔ best_model.pt</li>
<li>✔ training_curves.png</li>
<li>✔ confusion_matrix.png</li>
<li>✔ metrics.json</li>
</ul>

<h3>🖼️ Sample Prediction</h3>

<pre>
Image : sample_sneaker.png

Prediction

👟 Sneaker  → 77.47%
👡 Sandal   → 22.45%
👜 Bag      → 0.05%
</pre>

<hr>

<div align="center">

⭐ <b>If you like this project, don't forget to Star the repository!</b>

</div>
