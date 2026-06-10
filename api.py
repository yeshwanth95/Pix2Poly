import os
import json
import hashlib
import tempfile
from fastapi import FastAPI, UploadFile, HTTPException, Request, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
import uvicorn
from contextlib import asynccontextmanager
import base64
from typing import Optional
import requests
import shutil
from pathlib import Path
from diskcache import Cache
import re

from polygon_inference import PolygonInference
from utils import log

# API Key configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Get API key from environment variable
API_KEY = os.getenv("API_KEY")
MODEL_URL = os.getenv("MODEL_URL", "https://github.com/safelease/Pix2Poly/releases/download/main/runs_share.zip")

# Default model name (inria dataset)
DEFAULT_MODEL_NAME = "Pix2Poly_inria_coco_224"

# Cache configuration
CACHE_TTL = int(os.getenv("CACHE_TTL", 24 * 3600))  # 24 hours

# Global cache instance
cache = Cache(
    directory=os.path.join(tempfile.gettempdir(), "pix2poly_cache"),
    timeout=1,  # 1 second timeout for cache operations
    disk_min_file_size=0,  # Store all items on disk
    disk_pickle_protocol=4,  # Use protocol 4 for better compatibility
)

def get_cache_key(image_data: bytes, model_name: str = None, merge_tolerance: float = None, tile_overlap_ratio: float = None) -> str:
    """Generate a cache key from image data and parameters.
    
    Args:
        image_data: Raw image data
        model_name: Model name being used
        merge_tolerance: Merge tolerance parameter
        tile_overlap_ratio: Tile overlap ratio parameter
        
    Returns:
        SHA-256 hash of the image data combined with parameters as a string
    """
    image_hash = hashlib.sha256(image_data).hexdigest()
    return f"{image_hash}_{model_name}_{merge_tolerance}_{tile_overlap_ratio}"


def validate_model_name(model_name: str) -> bool:
    """Validate that the model name contains only safe characters.
    
    Args:
        model_name: The model name to validate
        
    Returns:
        True if the model name is valid, False otherwise
    """
    # Allow alphanumeric characters, underscores, and hyphens
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', model_name))


async def verify_api_key(
    header_key: Optional[str] = Depends(api_key_header),
    query_key: Optional[str] = Query(None, alias="api_key"),
) -> Optional[str]:
    """Verify the API key from either header or query parameter.

    If API authentication is not enabled (no API key configured),
    this function will always return None.

    Args:
        header_key: API key from X-API-Key header
        query_key: API key from api_key query parameter

    Returns:
        The verified API key or None if authentication is disabled

    Raises:
        HTTPException: 401 if API key is missing (when required)
        HTTPException: 403 if API key is invalid (when required)
    """
    if not API_KEY:
        return None

    api_key = header_key or query_key
    if not api_key:
        raise HTTPException(status_code=401, detail="API key is missing")
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


def download_model_files(model_url: str, target_dir: str) -> str:
    """Download model files to the target directory.

    Args:
        model_url: URL to download the model files from
        target_dir: Directory to save the model files to

    Returns:
        Path to the downloaded model directory

    Raises:
        ValueError: If download fails or model files are invalid
    """
    # Create target directory if it doesn't exist
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Check if model files already exist
    if target_path.exists() and any(target_path.iterdir()):
        log(f"Model files already exist in {target_dir}, skipping download", "INFO")
        return str(target_path)

    # Download the model files using requests
    zip_path = target_path / "runs_share.zip"
    
    log(f"Downloading model files from {model_url}", "INFO")
    response = requests.get(model_url, stream=True)
    response.raise_for_status()
    
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract the zip file
    log(f"Extracting model files to {target_dir}", "INFO")
    shutil.unpack_archive(zip_path, target_path)

    # Remove the zip file
    zip_path.unlink()

    return str(target_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the predictor on startup."""
    global model_dir
    
    # Download model files to a temporary directory
    model_dir = download_model_files(
        MODEL_URL,
        "/tmp/pix2poly_model",
    )

    # Initialize predictor with downloaded model using the default model name
    init_predictor(os.path.join(model_dir, "runs_share", DEFAULT_MODEL_NAME), DEFAULT_MODEL_NAME)
    yield


app = FastAPI(
    title="Polygon Inference API",
    description="API for inferring polygons in images using a trained model",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global predictor instance and current model tracking
predictor = None
current_model_name = None
model_dir = None


def init_predictor(experiment_path: str, model_name: str = None):
    """Initialize the global predictor instance."""
    global predictor, current_model_name
    if predictor is None or current_model_name != model_name:
        predictor = PolygonInference(experiment_path)
        current_model_name = model_name
        log(f"Loaded model: {model_name}", "INFO")


def load_model(model_name: str):
    """Load a specific model by name.
    
    Args:
        model_name: The name of the model to load (e.g., "Pix2Poly_inria_coco_224")
        
    Raises:
        HTTPException: If model name is invalid or model files don't exist
    """
    global predictor, current_model_name, model_dir

    log(f"Using model: {model_name}")
    
    if not validate_model_name(model_name):
        raise HTTPException(status_code=400, detail="Invalid model name. Only alphanumeric characters, underscores, and hyphens are allowed.")
    
    # Skip reloading if it's the same model
    if current_model_name == model_name and predictor is not None:
        return
    
    # Construct the full experiment path
    experiment_path = os.path.join(model_dir, "runs_share", model_name)
    
    # Check if the model directory exists
    if not os.path.exists(experiment_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found in downloaded model files")
    
    # Initialize the predictor with the new model
    init_predictor(experiment_path, model_name)


@app.post("/invocations")
async def invoke(
    request: Request,
    file: UploadFile = None,
    api_key: Optional[str] = Depends(verify_api_key),
    merge_tolerance: Optional[float] = Query(None, description="Tolerance for point-in-polygon tests during validation (in pixels, allows points to be slightly outside)"),
    tile_overlap_ratio: Optional[float] = Query(None, description="Overlap ratio between tiles (0.0 = no overlap, 1.0 = complete overlap)"),
    model_name: Optional[str] = Query(None, description="Name of the model to use (e.g., 'Pix2Poly_inria_coco_224')"),
):
    """Main inference endpoint for processing images.

    The endpoint accepts image data in three different ways:
    1. As a file upload using multipart/form-data (via the file parameter)
    2. As a base64-encoded image in a JSON payload with an 'image' field
    3. As raw image data in the request body

    Authentication can be provided in two ways:
    1. Via the X-API-Key header
    2. Via the api_key query parameter

    Configuration parameters can be provided in two ways:
    1. Via query parameters (merge_tolerance, tile_overlap_ratio, model_name)
    2. Via the JSON payload fields (merge_tolerance, tile_overlap_ratio, model_name)

    Args:
        request: The request containing the image data
        file: Optional uploaded file (multipart/form-data)
        api_key: Optional API key for authentication (required only if API key is configured)
        merge_tolerance: Optional tolerance for point-in-polygon tests during validation (in pixels, allows points to be slightly outside)
        tile_overlap_ratio: Optional overlap ratio between tiles (0.0 = no overlap, 1.0 = complete overlap)
        model_name: Optional name of the model to use (e.g., 'Pix2Poly_inria_coco_224')

    Returns:
        JSON response containing the inferred polygons

    Raises:
        HTTPException: 400 if no image data is found in the request
        HTTPException: 404 if the specified model is not found
        HTTPException: 500 if there is an error processing the image
        HTTPException: 401 if API key is missing (when API key is configured)
        HTTPException: 403 if API key is invalid (when API key is configured)
    """
    log(f"Invoking image analysis")

    # Initialize configuration parameters and validate ranges
    effective_merge_tolerance = merge_tolerance
    effective_tile_overlap_ratio = tile_overlap_ratio
    effective_model_name = model_name or DEFAULT_MODEL_NAME
    
    # Validate merge_tolerance (should be positive)
    if effective_merge_tolerance is not None and effective_merge_tolerance < 0:
        raise HTTPException(status_code=400, detail="merge_tolerance must be non-negative")
    
    # Validate tile_overlap_ratio (should be between 0.0 and 1.0)
    if effective_tile_overlap_ratio is not None and (effective_tile_overlap_ratio < 0.0 or effective_tile_overlap_ratio > 1.0):
        raise HTTPException(status_code=400, detail="tile_overlap_ratio must be between 0.0 and 1.0")

    if file:
        # Handle file upload
        image_data = await file.read()
    else:
        # Read request body
        body = await request.body()

        # Parse the request body
        try:
            data = json.loads(body)
            if "image" in data:
                # Handle base64 encoded image
                image_data = base64.b64decode(data["image"])
                
                # Extract configuration parameters from JSON (if query params not provided)
                if effective_merge_tolerance is None and "merge_tolerance" in data:
                    effective_merge_tolerance = float(data["merge_tolerance"])
                    if effective_merge_tolerance < 0:
                        raise HTTPException(status_code=400, detail="merge_tolerance must be non-negative")
                if effective_tile_overlap_ratio is None and "tile_overlap_ratio" in data:
                    effective_tile_overlap_ratio = float(data["tile_overlap_ratio"])
                    if effective_tile_overlap_ratio < 0.0 or effective_tile_overlap_ratio > 1.0:
                        raise HTTPException(status_code=400, detail="tile_overlap_ratio must be between 0.0 and 1.0")
                if model_name is None and "model_name" in data:
                    effective_model_name = str(data["model_name"])
            else:
                raise HTTPException(
                    status_code=400, detail="No image data found in request"
                )
        except json.JSONDecodeError:
            # Handle raw image data
            image_data = body

    # Load the requested model (this will only reload if it's different from the current model)
    load_model(effective_model_name)

    # Generate cache key including all configuration parameters
    cache_key = get_cache_key(image_data, effective_model_name, effective_merge_tolerance, effective_tile_overlap_ratio)
    cached_result = cache.get(cache_key)
    
    if cached_result is not None:
        return JSONResponse(content=cached_result)

    # Get inferences
    polygons = predictor.infer(image_data, merge_tolerance=effective_merge_tolerance, tile_overlap_ratio=effective_tile_overlap_ratio)

    # Prepare response
    response = {
        "polygons": polygons,
        "model_name": effective_model_name,
    }

    # Store result in cache
    cache.set(cache_key, response, expire=CACHE_TTL)

    return JSONResponse(content=response)

@app.get("/ping")
async def ping(api_key: Optional[str] = Depends(verify_api_key)):
    """Health check endpoint to verify service status."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.get("/clear-cache")
async def clear_cache(api_key: Optional[str] = Depends(verify_api_key)):
    """Clear the cache endpoint to remove all cached results."""
    cache.clear()
    return {"status": "success", "message": "Cache cleared successfully"}


if __name__ == "__main__":
    # Run the API
    uvicorn.run(app, host="0.0.0.0", port=8080)
