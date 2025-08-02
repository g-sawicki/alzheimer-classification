import logging
import os

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile

from backend.model_service import ModelService
from backend.schemas import (
    HealthCheck,
    PredictionResponse,
    ReloadModelRequest,
    ReloadModelResponse,
)
from backend.utils import PrometheusMiddleware, metrics, setting_otlp

MODEL_URI = "models:/alzheimer_classification/latest"

APP_NAME = os.environ["APP_NAME"]
EXPOSE_PORT = os.environ.get("EXPOSE_PORT", 8000)
OTLP_GRPC_ENDPOINT = os.environ.get("OTLP_GRPC_ENDPOINT", "tempo:4317")


app = FastAPI()
model_service = ModelService(MODEL_URI)

# Setting metrics middleware
app.add_middleware(PrometheusMiddleware, app_name=APP_NAME)
app.add_route("/metrics", metrics)

# Setting OpenTelemetry exporter
setting_otlp(app, APP_NAME, OTLP_GRPC_ENDPOINT)


class EndpointFilter(logging.Filter):
    # Uvicorn endpoint access log filter
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("GET /metrics") == -1


# Filter out /endpoint
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())


@app.get("/health-check")
def health_check() -> HealthCheck:
    return HealthCheck(status="OK")


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    logging.debug(f"Predicting an image.")
    if file.content_type not in ["image/png", "image/jpeg", "application/octet-stream"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    try:
        image_bytes = await file.read()
        prediction, confidence = model_service.predict(image_bytes)
        return PredictionResponse(prediction=prediction, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload-model")
def reload_model(request: ReloadModelRequest) -> ReloadModelResponse:
    logging.debug(f"Reloading model: {request.model_uri}")
    try:
        model_service.reload(request.model_uri)
        return ReloadModelResponse(
            detail=f"Model successfully reloaded from {request.model_uri}"
        )
    except Exception as e:
        logging.error(f"Reload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")


if __name__ == "__main__":
    # update uvicorn access logger format
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = (
        "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] [trace_id=%(otelTraceID)s span_id=%(otelSpanID)s resource.service.name=%(otelServiceName)s] - %(message)s"
    )
    uvicorn.run(app, host="0.0.0.0", port=EXPOSE_PORT, log_config=log_config)
