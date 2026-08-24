from pathlib import Path
from shutil import copyfile
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

root = Path(__file__).resolve().parents[1]
out = root / "screenshots"
html = (out / "_docker.html").resolve().as_uri()
opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--window-size=1400,900")
opts.add_argument("--force-device-scale-factor=1.25")
opts.add_argument("--hide-scrollbars")
driver = webdriver.Chrome(options=opts)
driver.get(html)
time.sleep(0.8)
el = driver.find_element(By.ID, "docker")
path = out / "docker.png"
el.screenshot(str(path))
print("saved", path, path.stat().st_size)
copyfile(path, out / "figure-2.11-docker.png")
driver.quit()
