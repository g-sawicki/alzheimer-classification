from pydantic import BaseModel


class HealthCheck(BaseModel):
    status: str = "OK"


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float


class ReloadModelRequest(BaseModel):
    model_uri: str

    model_config = {
        "json_schema_extra": {
            "example": {"model_uri": "models:/alzheimer_classification/latest"}
        }
    }


class ReloadModelResponse(BaseModel):
    detail: str
