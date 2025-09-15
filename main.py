import os
import shutil
import uuid
import torch
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from Model_Validated import EnhancedLungDiseasePredictor
from RAG import Rag


classes_idx = {
    "None": -1,
    "Bacterial Pneumonia": 0,
    "Tuberculosis": 1,
    "Normal": 2,
    "Covid-19": 3,
    "Viral Pneumonia": 4,
}
# Create FastAPI app
app = FastAPI()

rag = Rag()
rag.doc_precess()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save file to temp location
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_ext = os.path.splitext(file.filename)[-1]
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}{file_ext}")

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Test with an image
        predictor = EnhancedLungDiseasePredictor()
        result = predictor.predict_with_validation(temp_file_path)

        errors = result["validation_details"]["errors"]

        if errors:  # checks if the list is not empty
            return JSONResponse(
                status_code=400,
                content={
                    "message": f"Validation failed. {errors[0]}",
                    "data": None,
                    # {
                    #     "meta_data": result["validation_details"],
                    #     "image_validation": result["validation_passed"],
                    #     "predicted_class_index": classes_idx.get(result["prediction"]),
                    #     "predicted_label": result["prediction"],
                    #     "confidence": result["confidence"],
                    # },
                },
            )
        return JSONResponse(
            status_code=200,
            content={
                "message": "Prediction successful",
                "data": {
                    "meta_data": result["validation_details"],
                    "image_validation": result["validation_passed"],
                    "predicted_class_index": classes_idx.get(result["prediction"]),
                    "predicted_label": result["prediction"],
                    "confidence": result["confidence"],
                },
            },
        )
    except Exception as e:
        print(e)
        return JSONResponse(status_code=500, content={"message": str(e), "data": None})
    finally:
        # Cleanup the file
        os.remove(temp_file_path)


@app.get("/")
async def main():
    return "welcome to the x-ray backend api "


@app.post("/query")
async def query(request: Request):
    body = await request.json()  # ✅ Await the JSON body
    query_text = body.get("query")  # ✅ Extract the "query" from JSON
    print(query_text)
    print(body)
    if not query_text or query_text.strip() == "":
        return JSONResponse(status_code=200, content={"message": "Query not found"})

    # Assuming this function takes the query as input
    response = rag.rag_answer_generator(query_text)
    print("\nResponse:\n", response)

    return JSONResponse(status_code=200, content={"response": response})


# Custom handler for 404 errors
@app.exception_handler(StarletteHTTPException)
async def custom_404_handler(request, exc):
    if exc.status_code == 404:
        return JSONResponse(
            status_code=404,
            content={"error": "The requested route does not exist."},
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "An unexpected error occurred."},
    )
