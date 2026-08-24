# Cognevance Internship

This folder contains three internship tasks: customer churn prediction with machine learning, a Fashion-MNIST image classifier, and a Flask API that serves the churn model.

| Folder | Task | Summary |
| --- | --- | --- |
| [Task1](Task1/) | Customer Churn Prediction | Train and compare ML models on the Telco churn dataset |
| [task2](task2/) | Fashion-MNIST Image Classifier | Transfer learning with ResNet-18 in PyTorch |
| [task3](task3/) | Churn Prediction API | Flask web app and REST API with Docker support |

Each task has its own `README.md` with setup steps, project structure, and how to run it.

## Requirements

- Python 3.11 or 3.12
- Task 2 also needs PyTorch
- Task 3 can run locally or with Docker

## Quick start

### Task 1 — Customer Churn Prediction

```text
cd Task1
pip install -r requirements.txt
python src/run_pipeline.py
```

On Windows you can also run `run.bat`. Full details: [Task1/README.md](Task1/README.md).

### Task 2 — Fashion-MNIST Image Classifier

```text
cd task2
pip install -r requirements.txt
python train.py
python inference.py --image outputs/sample_sneaker.png --model outputs/best_model.pt
```

On Windows you can also run `setup.bat`, `train.bat`, and `run.bat`. Full details: [task2/README.md](task2/README.md).

### Task 3 — Customer Churn Prediction API

```text
cd task3
run.bat
```

Then open `http://127.0.0.1:8000`. With Docker:

```text
docker compose up --build
```

Full details: [task3/README.md](task3/README.md).
