# Demo — Flask Fashion Classifier API

## Health

```powershell
curl.exe -s http://localhost:8000/health
```

Expected:

```json
{"status":"ok","model_loaded":true,"test_accuracy":0.9896,"num_classes":5}
```

## Predict

```powershell
curl.exe -s -X POST http://localhost:8000/predict -F "file=@examples/samples/sample_sneaker.png"
```

## Visual demo

Open `examples/demo.html` while the API is running.
