import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image, ImageStat
from transformers import ViTImageProcessor, ViTForImageClassification
import torchvision.transforms as transforms
from scipy import ndimage
import warnings
from typing import Tuple, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import os
import logging
from typing import Tuple, Dict, Any
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import os
import logging
from typing import Tuple, Dict, Any
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XRayImageValidator:
    """
    Validator for detecting if a given image is a chest X-ray using heuristic + deep learning.
    """

    def __init__(self):
        self.min_width = 224
        self.min_height = 224
        self.max_width = 2048
        self.max_height = 2048
        self.allowed_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".dicom"]
        self.anatomical_validator = self._load_anatomical_validator()

    def _load_anatomical_validator(self):
        """Load a simple binary classifier to distinguish X-ray images"""
        try:
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, 2)
            model.load_state_dict(torch.load("xray_classifier.pth", map_location="cpu"))
            model.eval()
            logger.info("Anatomical validator model loaded successfully.")
            return model
        except Exception as e:
            logger.warning(f"Could not load anatomical validator model: {e}")
            return None

    def validate_file_format(self, image_path: str) -> bool:
        ext = os.path.splitext(image_path)[-1].lower()
        return ext in self.allowed_formats

    def validate_image_properties(self, image: Image.Image) -> bool:
        width, height = image.size
        aspect_ratio = width / height
        return (
            self.min_width <= width <= self.max_width
            and self.min_height <= height <= self.max_height
            and 0.5 <= aspect_ratio <= 2.0
        )

    def validate_grayscale_characteristics(self, image: Image.Image, errors: list) -> bool:
        gray = image.convert("L")
        img_array = np.array(gray)
        hist = cv2.calcHist([img_array], [0], None, [256], [0, 256])
        dark_ratio = np.sum(hist[:100]) / img_array.size
        contrast = np.std(img_array)

        logger.debug(f"Grayscale Check - Dark Ratio: {dark_ratio:.3f}, Contrast: {contrast:.2f}")

        if dark_ratio < 0.1:
            errors.append(f"Low dark ratio ({dark_ratio:.2f}): May not be an X-ray.")
            return False
        if contrast < 10:
            errors.append(f"Low contrast ({contrast:.2f}): May be poor-quality X-ray.")
            return False

        return True

    def validate_anatomical_structure(self, image: Image.Image, errors: list) -> bool:
        if self.anatomical_validator:
            preprocess = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            if image.mode != "RGB":
                image = image.convert("RGB")
            input_tensor = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                output = self.anatomical_validator(input_tensor)
                pred = torch.argmax(output, dim=1).item()
                logger.debug(f"Anatomical Validator Prediction: {pred}")
                if pred != 1:
                    errors.append("Anatomical model predicts this is not an X-ray.")
                    return False
                return True
        else:
            logger.warning("Anatomical validation skipped: model not loaded.")
            return True

    def comprehensive_validation(self, image_path: str) -> Tuple[bool, Dict[str, Any]]:
        results = {
            "file_format": False,
            "image_properties": False,
            "grayscale_characteristics": False,
            "anatomical_structure": False,
            "overall_valid": False,
            "errors": [],
        }

        try:
            if not self.validate_file_format(image_path):
                results["errors"].append("Unsupported file format.")
                raise ValueError("Unsupported file format.")
            results["file_format"] = True

            image = Image.open(image_path)

            if self.validate_image_properties(image):
                results["image_properties"] = True
            else:
                results["image_properties"] = False
                # results["errors"].append("Invalid image size or aspect ratio.")

            if self.validate_grayscale_characteristics(image, results["errors"]):
                results["grayscale_characteristics"] = True

            if self.validate_anatomical_structure(image, results["errors"]):
                results["anatomical_structure"] = True

            # Consider valid if 3 or more checks passed
            passed_checks = sum(
                [
                    results["file_format"],
                    results["image_properties"],
                    results["grayscale_characteristics"],
                    results["anatomical_structure"],
                ]
            )
            results["overall_valid"] = passed_checks >= 3

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            if str(e) not in results["errors"]:
                results["errors"].append(str(e))

        return results["overall_valid"], results


class EnhancedLungDiseasePredictor:
    """
    Enhanced lung disease predictor with built-in X-ray validation
    """

    def __init__(self, model_path: str = "./Lung Disease Dataset"):
        self.validator = XRayImageValidator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and feature extractor
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_path)
        self.model = ViTForImageClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Define label mapping based on your dataset
        self.label_map = {
            0: "Bacterial Pneumonia",
            1: "Tuberculosis",
            2: "Normal",
            3: "Covid-19",
            4: "Viral Pneumonia",
        }

    def predict_with_validation(self, image_path: str) -> Dict[str, Any]:
        """
        Predict lung disease with comprehensive validation
        """
        result = {
            "prediction": None,
            "confidence": None,
            "validation_passed": False,
            "validation_details": None,
            "error": None,
        }

        try:
            # Step 1: Comprehensive validation
            is_valid, validation_details = self.validator.comprehensive_validation(
                image_path
            )
            result["validation_details"] = validation_details

            if not is_valid:
                result["error"] = (
                    "Image failed validation checks. This does not appear to be a valid chest X-ray image."
                )
                return result

            result["validation_passed"] = True

            # Step 2: Make prediction
            image = Image.open(image_path).convert("RGB")
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_class = probabilities.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()

            result["prediction"] = self.label_map[predicted_class]
            result["confidence"] = confidence

        except Exception as e:
            print(e)
            result["error"] = f"Prediction failed: {str(e)}"
            logger.error(f"Prediction error: {e}")

        return result

    def batch_predict_with_validation(self, image_paths: list) -> list:
        """
        Predict multiple images with validation
        """
        results = []
        for image_path in image_paths:
            result = self.predict_with_validation(image_path)
            results.append({"image_path": image_path, **result})
        return results


# Enhanced prediction function with validation
def predict_lung_disease_with_validation(
    image_path: str, model_path: str = "./Lung Disease Dataset"
):
    """
    Enhanced prediction function that validates X-ray images before prediction
    """
    predictor = EnhancedLungDiseasePredictor(model_path)
    return predictor.predict_with_validation(image_path)


# Example usage
if __name__ == "__main__":
    # Initialize the enhanced predictor
    predictor = EnhancedLungDiseasePredictor()

    # Test with an image
    image_path = "144.jpeg"  # Replace with your image path

    result = predictor.predict_with_validation(image_path)

    print("=== VALIDATION AND PREDICTION RESULTS ===")
    print(f"Image Path: {image_path}")
    print(f"Validation Passed: {result['validation_passed']}")

    if result["validation_passed"]:
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
    else:
        print(f"Error: {result['error']}")

    print("\n=== DETAILED VALIDATION RESULTS ===")
    print(result["validation_details"].items())
    for check, passed in result["validation_details"].items():
        if check != "errors":
            print(f"{check.replace('_', ' ').title()}: {'✓' if passed else '✗'}")

    if result["validation_details"]["errors"]:
        print("\nValidation Errors:")
        for error in result["validation_details"]["errors"]:
            print(f"- {error}")
