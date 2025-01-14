from fastapi import FastAPI, File, UploadFile
from keras.models import load_model
from keras.utils import register_keras_serializable
from keras.losses import MeanSquaredError
import io
import tempfile

app = FastAPI()

# Register custom loss function
@register_keras_serializable()
def mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)

# Initialize the model variable
model = None

# Function to load the model from the uploaded file
def load_model_from_file(model_file: UploadFile):
    global model
    # Create a temporary file to save the uploaded model content
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        # Write the uploaded file content to the temporary file
        temp_file.write(model_file.file.read())
        temp_file_path = temp_file.name  # Get the file path

    # Load the model using Keras' load_model function (for .h5 files)
    model = load_model(temp_file_path, custom_objects={'mse': mse})
    print("Model loaded successfully.")

@app.post("/upload_model")
async def upload_model(model_file: UploadFile = File(...)):
    # Load the model from the uploaded file
    load_model_from_file(model_file)
    return {"message": "Model uploaded and loaded successfully."}

@app.get("/")
async def root():
    return {"message": "Welcome to my FastAPI app!"}
