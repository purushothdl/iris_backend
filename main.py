from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from iris_model import iris_predict
from fastapi.middleware.cors import CORSMiddleware

# Create a FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. You can specify a list of allowed origins here.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.). You can specify a list of allowed methods here.
    allow_headers=["*"],  # Allows all headers. You can specify a list of allowed headers here.
)

# Define the request body using Pydantic
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define the prediction endpoint
@app.post("/predict")
async def predict(iris: IrisData):
    # Convert the input data to the format expected by the model
    input_data = np.array([iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width])
    
    # Make a prediction
    prediction = iris_predict(input_data)

    # Return the prediction result
    return {"flower_type": prediction}
