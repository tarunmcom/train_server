import os
import subprocess
import uuid
import logging
import time
from datetime import datetime, timedelta
from threading import Thread
from collections import defaultdict
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import glob
import re
import json
import requests
import shutil
from autotrain.trainers.dreambooth.train_xl import main, TrainingState
import argparse
from queue import Queue
import threading
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, UUID4, constr
from typing import Optional, Dict, Any
import uvicorn
from pydantic import BaseModel, UUID4, Field, constr

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# Global variables
jobs = defaultdict(dict)
executor = ThreadPoolExecutor(max_workers=1)  # Only allow one training job at a time

# Configuration
DEFAULT_CONFIG = {
    "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
    "prompt": "photo of a person",
    "push_to_hub": False,
    "hf_token": "your_huggingface_token_here",
    "hf_username": "your_huggingface_username_here",
    "learning_rate": 1e-4,
    "num_steps": 500,
    "batch_size": 1,
    "gradient_accumulation": 4,
    "resolution": 1024,
    "use_8bit_adam": True,
    "use_xformers": True,
    "mixed_precision": "fp16",
    "train_text_encoder": False,
    "disable_gradient_checkpointing": False,
    "callback_url": "http://example.com/callback",
}

# Use AWS params from env vars
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-2')

# Add this near the top of your file, after the imports
if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION]):
    raise EnvironmentError("AWS credentials are not properly set in environment variables.")

# Add this class near the top of the file
class JobTrainingState:
    def __init__(self):
        self._lock = threading.Lock()
        self._is_training = False
        self._current_step = 0
        self._total_steps = 0
        self._current_loss = 0.0
        self._progress_percentage = 0.0
        self.training_state = None
        self.last_update_time = None

    def update_state(self, training_state):
        with self._lock:
            if training_state:
                self._current_step = training_state.current_step
                self._current_loss = training_state.current_loss if training_state.current_loss is not None else 0.0
                if self._total_steps > 0:
                    self._progress_percentage = (self._current_step / self._total_steps) * 100
            self.training_state = training_state
            self.last_update_time = datetime.now()

    @property
    def current_step(self):
        with self._lock:
            return self._current_step

    @property
    def current_loss(self):
        with self._lock:
            return self._current_loss

    @property
    def progress_percentage(self):
        with self._lock:
            return self._progress_percentage

    @property
    def is_training(self):
        with self._lock:
            return self._is_training

    @is_training.setter
    def is_training(self, value):
        with self._lock:
            self._is_training = value

    @property
    def total_steps(self):
        with self._lock:
            return self._total_steps

    @total_steps.setter
    def total_steps(self, value):
        with self._lock:
            self._total_steps = value

def upload_safetensor_to_s3(job_id, unique_user_id, project_name, training_args):
    try:
        # Create a boto3 client
        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        s3 = session.client('s3')

        # Find the generated .safetensor file
        safetensor_files = glob.glob(f"{project_name}/*.safetensors")
        if not safetensor_files:
            logging.error(f"No .safetensor file found for job {job_id}")
            return False, None

        safetensor_file = safetensor_files[0]
        safetensor_name = os.path.basename(safetensor_file)

        # Create the folder structure in the S3 bucket if it doesn't exist
        s3.put_object(Bucket="myloras", Key=f"{unique_user_id}/{project_name}/")

        # Upload the safetensor file to S3 with the job_id as the filename
        s3_key = f"{unique_user_id}/{project_name}/{job_id}.safetensors"
        s3.upload_file(safetensor_file, "myloras", s3_key)

        # Add this line to store the full S3 path
        full_s3_path = f"s3://myloras/{s3_key}"

        # Create metadata JSON
        metadata = {
            "job_id": job_id,
            "unique_user_id": str(unique_user_id),
            "project_name": project_name,
            "person_name": training_args["person_name"],
            "training_params": {
                "model_name": training_args["model_name"],
                "prompt": training_args["prompt"],
                "learning_rate": training_args["learning_rate"],
                "num_steps": training_args["num_steps"],
                "batch_size": training_args["batch_size"],
                "gradient_accumulation": training_args["gradient_accumulation"],
                "resolution": training_args["resolution"],
                "use_8bit_adam": training_args["use_8bit_adam"],
                "use_xformers": training_args["use_xformers"],
                "mixed_precision": training_args["mixed_precision"],
                "train_text_encoder": training_args["train_text_encoder"],
                "disable_gradient_checkpointing": training_args["disable_gradient_checkpointing"]
            },
            "timestamp": datetime.now().isoformat()
        }

        # Upload metadata JSON to S3
        metadata_json = json.dumps(metadata, indent=2)
        metadata_key = f"{unique_user_id}/{project_name}/{job_id}_metadata.json"
        s3.put_object(Body=metadata_json, Bucket="myloras", Key=metadata_key)

        logging.info(f"Uploaded {s3_key} and metadata to S3 bucket 'myloras' for job {job_id}")
        return True, full_s3_path  # Return the full S3 path along with success status

    except ClientError as e:
        logging.error(f"Error uploading files to S3: {e}")
        return False, None
    except NoCredentialsError:
        logging.error("AWS credentials not found or invalid")
        return False, None

def download_s3_images(bucket_name, s3_folder, local_dir=None):
    try:
        # Create a boto3 client
        session = boto3.Session(
           aws_access_key_id=AWS_ACCESS_KEY_ID,
           aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
           region_name=AWS_REGION  )
        s3 = session.client('s3')

        # Create a random folder name if not provided
        if local_dir is None:
            local_dir = str(uuid.uuid4())

        # Create the local directory
        os.makedirs(local_dir, exist_ok=True)

        # List objects within the S3 folder
        result = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)
        if 'Contents' not in result:
            logging.info(f"No objects found in {s3_folder}")
            return

        # Download each object
        for obj in result['Contents']:
            # Get the file path
            s3_file = obj['Key']
            
            # Skip if it's a folder or if it's in the "_thumbs" folder
            if s3_file.endswith('/') or f'{s3_folder}_thumbs' in s3_file:
                continue
            
            # Remove the folder name from the file path
            local_file = s3_file.replace(s3_folder, '', 1).lstrip('/')
            local_file_path = os.path.join(local_dir, local_file)

            # Create directories if they don't exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the file
            logging.info(f"Downloading {s3_file} to {local_file_path}")
            s3.download_file(bucket_name, s3_file, local_file_path)

        logging.info(f"All files downloaded to {local_dir}")
        return True

    except ClientError as e:
        logging.error(f"Error downloading files: {e}")
        return False
    except NoCredentialsError:
        logging.error("AWS credentials not found or invalid")
        return False

def call_callback_endpoint(job_id, project_name, s3_bucket, s3_folder, person_name, status, datetime, reason=None):
    callback_url = DEFAULT_CONFIG["callback_url"]
    payload = {
        "job_id": str(job_id),  # Convert to string if it's a UUID
        "project_name": project_name,
        "s3_bucket": s3_bucket,
        "s3_folder": s3_folder,
        "person_name": person_name,
        "status": status,
        "datetime": datetime,
        "reason": reason
    }
    try:
        # Convert any UUID objects in the payload to strings
        payload = {k: str(v) if isinstance(v, uuid.UUID) else v for k, v in payload.items()}
        response = requests.post(callback_url, json=payload)
        response.raise_for_status()
        logging.info(f"Callback sent successfully for job {job_id}")
    except requests.RequestException as e:
        logging.error(f"Failed to send callback for job {job_id}: {str(e)}")

def setup_training_args(args):
    training_args = argparse.Namespace()
    
    # Essential parameters
    training_args.pretrained_model_name_or_path = args["model_name"]
    training_args.pretrained_vae_model_name_or_path = None
    training_args.instance_prompt = args["prompt"]
    training_args.output_dir = args["project_name"]
    training_args.instance_data_dir = f"images_{args['job_id']}"
    
    # Core training parameters
    training_args.learning_rate = args["learning_rate"]
    training_args.max_train_steps = args["num_steps"]
    training_args.train_batch_size = args["batch_size"]
    training_args.gradient_accumulation_steps = args["gradient_accumulation"]
    training_args.resolution = args["resolution"]
    training_args.use_8bit_adam = args["use_8bit_adam"]
    training_args.enable_xformers_memory_efficient_attention = args["use_xformers"]
    training_args.mixed_precision = args["mixed_precision"]
    training_args.train_text_encoder = args["train_text_encoder"]
    training_args.gradient_checkpointing = not args["disable_gradient_checkpointing"]
    
    # Additional parameters with default values
    training_args.revision = None
    training_args.variant = None
    training_args.with_prior_preservation = False
    training_args.num_class_images = 50
    training_args.class_data_dir = None
    training_args.class_prompt = None
    training_args.seed = None
    training_args.center_crop = False
    training_args.random_flip = False
    training_args.adam_beta1 = 0.9
    training_args.adam_beta2 = 0.999
    training_args.adam_weight_decay = 1e-2
    training_args.adam_weight_decay_text_encoder = 1e-2
    training_args.text_encoder_lr = None
    training_args.adam_epsilon = 1e-08
    training_args.max_grad_norm = 1.0
    training_args.allow_tf32 = False
    training_args.dataloader_num_workers = 0
    training_args.num_validation_images = 4
    training_args.validation_epochs = 50
    training_args.checkpointing_steps = 500
    training_args.checkpoints_total_limit = None
    training_args.resume_from_checkpoint = None
    training_args.enable_cpu_offload = False
    training_args.scale_lr = False
    training_args.lr_scheduler = "constant"
    training_args.lr_warmup_steps = 0
    training_args.lr_num_cycles = 1
    training_args.lr_power = 1.0
    training_args.rank = 4
    training_args.validation_prompt = None
    training_args.num_train_epochs = None
    training_args.report_to = "tensorboard"  # Changed from "tensorboard" to "none"
    training_args.logging_dir = "logs"
    training_args.optimizer = "adamw"
    training_args.snr_gamma = None
    training_args.use_dora = False
    training_args.do_edm_style_training = False
    training_args.repeats = 1
    training_args.prior_loss_weight = 1.0
    training_args.sample_batch_size = 4
    training_args.prodigy_beta3 = 0.999
    training_args.prodigy_decouple = True
    training_args.prodigy_use_bias_correction = True
    training_args.prodigy_safeguard_warmup = True
    
    # Hub related parameters
    if args["push_to_hub"]:
        training_args.push_to_hub = True
        training_args.hub_token = args["hf_token"]
        training_args.hub_model_id = f"{args['hf_username']}/{args['project_name']}"
    else:
        training_args.push_to_hub = False
        training_args.hub_token = None
        training_args.hub_model_id = None
    
    return training_args

def train_lora(job_id, args):
    logging.info(f"Starting training for job {job_id}")
    jobs[job_id]["status"] = "BUSY"
    jobs[job_id]["stage"] = "downloading"
    jobs[job_id]["message"] = "Downloading images from S3"
    jobs[job_id]["start_time"] = datetime.now()
    
    # Download images from S3
    local_image_folder = f"images_{job_id}"
    s3_download_success = download_s3_images(args["s3_bucket"], args["s3_folder"], local_image_folder)
    
    if not s3_download_success:
        jobs[job_id]["status"] = "FAILED"
        jobs[job_id]["message"] = "Failed to download images from S3"
        jobs[job_id]["stage"] = "download_failed"
        return
    
    jobs[job_id]["status"] = "BUSY"
    jobs[job_id]["stage"] = "training"
    jobs[job_id]["message"] = "Training in progress"
    
    try:
        # Set up training arguments
        training_args = setup_training_args(args)
        
        # Initialize training state with thread-safe implementation
        training_state = TrainingState()
        jobs[job_id]["training_state"] = JobTrainingState()
        jobs[job_id]["training_state"].is_training = True
        jobs[job_id]["training_state"]._total_steps = args["num_steps"]
        jobs[job_id]["last_update"] = datetime.now()

        # Create a callback function to update the training state
        def update_callback(state):
            if job_id in jobs and "training_state" in jobs[job_id]:
                jobs[job_id]["training_state"].update_state(state)
                jobs[job_id]["last_update"] = datetime.now()

        # Start training with callback
        main(training_args, training_state=training_state, callback=update_callback)
        
        # Upload the generated .safetensor file and metadata to S3
        upload_success, full_s3_path = upload_safetensor_to_s3(job_id, args["unique_user_id"], args["project_name"], args)
        
        if upload_success:
            jobs[job_id]["status"] = "COMPLETED"
            jobs[job_id]["message"] = f"Training completed successfully. Safetensor file and metadata uploaded to S3. Full path: {full_s3_path}"
            jobs[job_id]["stage"] = "completed"
            jobs[job_id]["s3_upload_success"] = True
            jobs[job_id]["safetensor_path"] = full_s3_path
            logging.info(f"Job {job_id} completed successfully and safetensor with metadata uploaded to {full_s3_path}")
            call_callback_endpoint(job_id, args["project_name"], args["s3_bucket"], args["s3_folder"], 
                                   args["person_name"], "COMPLETED", datetime.now().isoformat())
        else:
            jobs[job_id]["status"] = "FAILED"
            jobs[job_id]["message"] = "Training completed, but failed to upload safetensor file and metadata to S3."
            jobs[job_id]["stage"] = "upload_failed"
            logging.error(f"Job {job_id} training completed, but failed to upload safetensor and metadata")
            call_callback_endpoint(job_id, args["project_name"], args["s3_bucket"], args["s3_folder"], 
                                   args["person_name"], "FAILED", datetime.now().isoformat(), 
                                   "Failed to upload safetensor file and metadata to S3")
    except Exception as e:
        jobs[job_id]["status"] = "FAILED"
        jobs[job_id]["message"] = f"An error occurred: {str(e)}"
        jobs[job_id]["stage"] = "error"
        logging.exception(f"An error occurred in job {job_id}")
        call_callback_endpoint(job_id, args["project_name"], args["s3_bucket"], args["s3_folder"], 
                               args["person_name"], "FAILED", datetime.now().isoformat(), str(e))
    finally:
        if "training_state" in jobs[job_id]:
            jobs[job_id]["training_state"].is_training = False
        # Clean up the local image folder
        if os.path.exists(local_image_folder):
            shutil.rmtree(local_image_folder)
        logging.info(f"Cleaned up local image folder for job {job_id}")

        # Clean up the project folder
        project_folder = args["project_name"]
        if os.path.exists(project_folder):
            shutil.rmtree(project_folder)
        logging.info(f"Cleaned up project folder for job {job_id}")

# Pydantic models for request validation
class TrainingRequest(BaseModel):
    class Config:
        protected_namespaces = ()  # Add this to disable protected namespace warnings
    
    job_id: str
    unique_user_id: UUID4
    project_name: str = Field(..., pattern=r'^[a-zA-Z0-9_-]{3,63}$')
    s3_bucket: str = Field(..., pattern=r'^[a-z0-9.-]{3,63}$')
    s3_folder: str = Field(..., pattern=r'^[a-zA-Z0-9_/-]{1,1024}$')
    person_name: str
    model_name: Optional[str] = DEFAULT_CONFIG["model_name"]
    push_to_hub: Optional[bool] = DEFAULT_CONFIG["push_to_hub"]
    hf_token: Optional[str] = DEFAULT_CONFIG["hf_token"]
    hf_username: Optional[str] = DEFAULT_CONFIG["hf_username"]
    learning_rate: Optional[float] = DEFAULT_CONFIG["learning_rate"]
    num_steps: Optional[int] = DEFAULT_CONFIG["num_steps"]
    batch_size: Optional[int] = DEFAULT_CONFIG["batch_size"]
    gradient_accumulation: Optional[int] = DEFAULT_CONFIG["gradient_accumulation"]
    resolution: Optional[int] = DEFAULT_CONFIG["resolution"]
    use_8bit_adam: Optional[bool] = DEFAULT_CONFIG["use_8bit_adam"]
    use_xformers: Optional[bool] = DEFAULT_CONFIG["use_xformers"]
    mixed_precision: Optional[str] = DEFAULT_CONFIG["mixed_precision"]
    train_text_encoder: Optional[bool] = DEFAULT_CONFIG["train_text_encoder"]
    disable_gradient_checkpointing: Optional[bool] = DEFAULT_CONFIG["disable_gradient_checkpointing"]
    callback_url: Optional[str] = DEFAULT_CONFIG["callback_url"]

class JobStatus(BaseModel):
    job_id: str
    status: str
    message: str
    stage: str
    safetensor_path: str = ""
    steps_completed: Optional[int] = None
    total_steps: Optional[int] = None
    current_loss: Optional[float] = None
    progress_percentage: Optional[float] = None
    is_training: Optional[bool] = None
    elapsed_time: Optional[str] = None
    estimated_completion_time: Optional[str] = None

def is_server_busy():
    return any(job.get("status") == "BUSY" for job in jobs.values())

@app.get("/")
async def root():
    return {"message": "alive"}

@app.post("/train", status_code=202)
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    if is_server_busy():
        raise HTTPException(
            status_code=409,
            detail="Another job is already running. Please wait for it to complete."
        )
    
    if request.job_id in jobs:
        raise HTTPException(
            status_code=409,
            detail=f"Job {request.job_id} already exists"
        )

    # Convert Pydantic model to dict and merge with DEFAULT_CONFIG
    training_args = DEFAULT_CONFIG.copy()
    training_args.update(request.dict())
    
    # Update prompt with person_name
    training_args["prompt"] = f"photo of {request.person_name}"
    
    # Initialize job status
    jobs[request.job_id] = {
        "status": "INITIALIZING",
        "message": "Job is being set up",
        "steps_completed": 0,
        "num_steps": training_args["num_steps"],
        "stage": "initializing"
    }
    
    # Clean up existing project folder
    if os.path.exists(training_args["project_name"]):
        shutil.rmtree(training_args["project_name"])
        logging.info(f"Cleaned up existing project folder: {training_args['project_name']}")

    # Submit training job to background tasks
    background_tasks.add_task(train_lora, request.job_id, training_args)
    
    return {"message": "Training job initiated", "job_id": request.job_id}

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    if not job_id:
        raise HTTPException(status_code=400, detail="Invalid job_id")
        
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = jobs[job_id]
    
    # Check for stalled training
    if (job_info.get("status") == "BUSY" and 
        "last_update" in job_info and 
        datetime.now() - job_info["last_update"] > timedelta(minutes=55)):
        job_info["status"] = "STALLED"
        job_info["message"] = "Training appears to be stalled"

    status_info = {
        "job_id": job_id,
        "status": job_info.get("status", "UNKNOWN"),
        "message": job_info.get("message", "Status unknown"),
        "stage": job_info.get("stage", "unknown"),
        "safetensor_path": job_info.get("safetensor_path", "")
    }
    
    # Add training metrics if available
    try:
        if ("training_state" in job_info and 
            job_info["training_state"] is not None and 
            job_info["training_state"].training_state is not None):
            
            job_training_state = job_info["training_state"]
            training_state = job_training_state.training_state
            
            try:
                current_step = int(training_state.current_step)
                total_steps = int(job_training_state.total_steps)
                current_loss = float(training_state.current_loss) if training_state.current_loss is not None else 0.0
                
                # Calculate progress
                progress_percentage = 0.0
                if total_steps > 0:
                    progress_percentage = (current_step / total_steps) * 100
                    progress_percentage = min(100.0, max(0.0, progress_percentage))
                
                # Calculate elapsed time
                elapsed_time = datetime.now() - job_info["start_time"]
                estimated_completion_time = None
                if current_step > 0:
                    seconds_per_step = elapsed_time.total_seconds() / current_step
                    remaining_steps = total_steps - current_step
                    estimated_completion_time = datetime.now() + timedelta(seconds=remaining_steps * seconds_per_step)

                status_info.update({
                    "steps_completed": current_step,
                    "total_steps": total_steps,
                    "current_loss": round(current_loss, 4),
                    "progress_percentage": round(progress_percentage, 2),
                    "is_training": bool(job_training_state.is_training),
                    "elapsed_time": str(elapsed_time),
                    "estimated_completion_time": estimated_completion_time.isoformat() if estimated_completion_time else None
                })

            except (ValueError, TypeError) as e:
                logging.error(f"Error converting training values for job {job_id}: {str(e)}")
                status_info.update({
                    "steps_completed": 0,
                    "total_steps": 0,
                    "current_loss": 0.0,
                    "progress_percentage": 0.0,
                    "is_training": False,
                    "training_error": "Error reading training values"
                })
    except Exception as e:
        logging.error(f"Error processing training state for job {job_id}: {str(e)}")
        status_info["training_error"] = "Error processing training state"
    
    return status_info

@app.get("/jobs")
async def list_jobs():
    return {
        job_id: {
            "status": job_info["status"],
            "message": job_info.get("message", "")
        } for job_id, job_info in jobs.items()
    }

@app.get("/busy")
async def check_server_busy():
    return {"busy": is_server_busy()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
