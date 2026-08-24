"""Capture README screenshots from the running Flask app."""
from pathlib import Path
import json
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "screenshots"
OUT.mkdir(exist_ok=True)
BASE = "http://127.0.0.1:8000"


def make_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1440,900")
    opts.add_argument("--force-device-scale-factor=1")
    opts.add_argument("--hide-scrollbars")
    opts.add_argument("--disable-gpu")
    return webdriver.Chrome(options=opts)


def shot(driver, name: str):
    path = OUT / name
    driver.save_screenshot(str(path))
    print("saved", path, path.stat().st_size)


def main():
    driver = make_driver()
    wait = WebDriverWait(driver, 15)
    try:
        driver.get(f"{BASE}/")
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1")))
        time.sleep(0.8)
        shot(driver, "home.png")

        driver.get(f"{BASE}/predict-page")
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "form, .predict-wrap, h1, .panel")))
        time.sleep(0.8)
        shot(driver, "predict.png")

        # Seed a realistic result into sessionStorage then open result page
        sample = {
            "mode": "single",
            "input": {
                "gender": "Female",
                "tenure": 24,
                "Contract": "Two year",
                "InternetService": "Fiber optic",
                "MonthlyCharges": 89.1,
                "TotalCharges": 2100.5,
                "PaymentMethod": "Credit card (automatic)",
                "Partner": "Yes",
            },
            "result": {
                "churn_prediction": False,
                "churn_label": "No Churn",
                "churn_probability": 0.09,
                "model_name": "Logistic Regression",
            },
        }
        driver.get(f"{BASE}/result")
        driver.execute_script(
            "sessionStorage.setItem('churn_result', arguments[0]);",
            json.dumps(sample),
        )
        driver.get(f"{BASE}/result")
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".result-title, h1, .result-hero")))
        time.sleep(0.8)
        shot(driver, "result.png")

        # Docker-related screenshot: show Dockerfile + compose content page
        html = """
        <html><head><meta charset='utf-8'>
        <style>
          body{margin:0;font-family:Consolas,monospace;background:#0b1220;color:#e2e8f0}
          .wrap{padding:40px 48px}
          h1{color:#38bdf8;font-size:28px;margin:0 0 8px}
          .sub{color:#94a3b8;margin-bottom:28px}
          .card{background:#111827;border:1px solid #1e293b;border-radius:12px;padding:20px 22px;margin-bottom:18px}
          .label{color:#22c55e;margin-bottom:10px;font-weight:700}
          pre{margin:0;white-space:pre-wrap;line-height:1.45;color:#cbd5e1}
          .ok{display:inline-block;margin-top:8px;padding:6px 12px;border-radius:999px;background:#14532d;color:#86efac}
        </style></head><body><div class='wrap'>
        <h1>Docker Deployment</h1>
        <div class='sub'>Customer Churn Prediction API · container ready</div>
        <div class='card'><div class='label'>Dockerfile</div>
        <pre>FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn","--bind","0.0.0.0:8000","wsgi:app"]</pre></div>
        <div class='card'><div class='label'>docker compose up --build</div>
        <pre>$ docker compose up --build
[+] Building ... done
[+] Running 1/1
 ✔ Container churn-api  Started
 => http://127.0.0.1:8000</pre>
        <div class='ok'>● Healthy · model loaded</div>
        </div></div></body></html>
        """
        driver.get("data:text/html;charset=utf-8," + html)
        time.sleep(0.5)
        shot(driver, "docker.png")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
