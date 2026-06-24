# Financial Table Classifier API

Simple API for classifying financial report table images as either:

- `balance`
- `activity`

The service is deployed on Google Cloud Run.

## Cloud Run URL

```text
https://table-classifier-api-293379362892.europe-west1.run.app
```

## API Endpoints

### 1. Health Check

Checks that the service is running.

```http
GET /health
```

Example:

```bash
curl https://table-classifier-api-293379362892.europe-west1.run.app/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "Financial Table Classifier API"
}
```

---

### 2. Model Info

Returns basic information about the loaded model.

```http
GET /model-info
```

Example:

```bash
curl https://table-classifier-api-293379362892.europe-west1.run.app/model-info
```

Example response:

```json
{
  "model_path": "models/best.pt",
  "task": "detect",
  "class_names": {
    "0": "account_opinion",
    "1": "column_clarification",
    "2": "column_last_year",
    "3": "column_text",
    "4": "column_this_year",
    "5": "table_activities",
    "6": "table_balances",
    "7": "text_this_year"
  },
  "num_classes": 8,
  "device": "cpu"
}
```

---

### 3. Predict

Uploads an image and returns the predicted table type.

```http
POST /predict
```

The request must include an image file in a form-data field named `file`.

Example:

```bash
curl.exe -X POST "https://table-classifier-api-293379362892.europe-west1.run.app/predict" ^
  -F "file=@samples/2024-10-27_09-28-29__117-2024-01244081_2_balance.jpg"
```

On Mac/Linux:

```bash
curl -X POST "https://table-classifier-api-293379362892.europe-west1.run.app/predict" \
  -F "file=@samples/2024-10-27_09-28-29__117-2024-01244081_2_balance.jpg"
```

Example response:

```json
{
  "filename": "2024-10-27_09-28-29__117-2024-01244081_2_balance.jpg",
  "prediction": "balance",
  "confidence": 0.9447,
  "model_class_name": "table_balances",
  "detections": [],
  "message": "Prediction completed successfully"
}
```

## Response Fields

| Field | Description |
|---|---|
| `filename` | Uploaded file name |
| `prediction` | Final business label: `balance` or `activity` |
| `confidence` | Confidence score of the selected table prediction |
| `model_class_name` | Original model class name, for example `table_balances` or `table_activities` |
| `detections` | Full list of detections returned by the model |
| `message` | Request status message |

## Notes

- The API uses a YOLO object detection model.
- The final prediction only uses table classes: `table_balances` and `table_activities`.
- Other detections such as columns or text areas may appear in the `detections` list, but they are not used as the final table prediction.
