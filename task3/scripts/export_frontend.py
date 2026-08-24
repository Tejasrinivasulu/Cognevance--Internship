from pathlib import Path

src = Path(__file__).resolve().parents[1] / "app" / "templates" / "landing.html"
out = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
html = src.read_text(encoding="utf-8")
html = html.replace("{{ url_for('static', filename='css/style.css') }}", "style.css")
html = html.replace("{{ url_for('static', filename='js/script.js') }}", "script.js")
out.write_text(html, encoding="utf-8")
print(f"Wrote {out}")
