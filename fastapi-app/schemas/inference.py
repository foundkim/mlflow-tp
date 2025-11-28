from pydantic import BaseModel


class InferenceIn(BaseModel):
    sepal_lenght: float
    sepal_width: float
    petal_lenght: float
    petal_width: float


class InferenceOut(BaseModel):
    prediction: list
