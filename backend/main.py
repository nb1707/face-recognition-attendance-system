from supabase import create_client
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json
from datetime import datetime
import shutil
from fastapi.responses import FileResponse
import csv
from io import BytesIO

# load environment variables from .env file
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/app")
def open_app():
    return FileResponse("static/index.html")

os.makedirs("storage/images", exist_ok=True)
os.makedirs("storage/embeddings", exist_ok=True)

# Store for unknown faces captured during recognition
os.makedirs("storage/unknown_faces", exist_ok=True)

# store video 
os.makedirs("storage/videos", exist_ok=True)

USERS_FILE = "storage/users.json"
ATTENDANCE_FILE = "storage/attendance_logs.json"
ATTENDANCE_CANDIDATE_FILE = "storage/attendance_candidates.json"

RECOGNITION_THRESHOLD = 0.45

BLUR_THRESHOLD = 50
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 220

ATTENDANCE_COOLDOWN_SECONDS = 60
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=-1)  # CPU mode

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Load users from local JSON file 
# def load_users():
#     if not os.path.exists(USERS_FILE):
#         return {}

#     with open(USERS_FILE, "r", encoding="utf-8") as f:
#         return json.load(f)

# Load users from Supabase
def load_users():
    response = supabase.table("users").select("*").execute()

    users = {}

    for user in response.data:
        users[user["user_id"]] = user

    return users

# def save_attendance_log(user_id, full_name, role, confidence, camera_location):
    logs = []

    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)

    now = datetime.now()

    for log in reversed(logs):
        if log["user_id"] == user_id:
            last_time = datetime.fromisoformat(log["timestamp"])
            diff = (now - last_time).total_seconds()

            if diff < ATTENDANCE_COOLDOWN_SECONDS:
                return False
            break

    logs.append({
        "user_id": user_id,
        "full_name": full_name,
        "role": role,
        "timestamp": now.isoformat(),
        "confidence": confidence,
        "camera_location": camera_location,
        "status": "present"
    })

    with open(ATTENDANCE_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)
    return True
def save_attendance_log(user_id, full_name, role, confidence, camera_id, camera_location, camera_type):
    now = datetime.now()

    response = (
        supabase
        .table("detection_logs")
        .select("*")
        .eq("user_id", user_id)
        .order("timestamp", desc=True)
        .limit(1)
        .execute()
    )

    if len(response.data) > 0:
        last_time = datetime.fromisoformat(response.data[0]["timestamp"])
        diff = (now - last_time).total_seconds()

        if diff < ATTENDANCE_COOLDOWN_SECONDS:
            return False

    data = {
        "user_id": user_id,
        "full_name": full_name,
        "role": role,
        "timestamp": now.isoformat(),
        "confidence": confidence,
        "camera_id": camera_id,
        "camera_location": camera_location,
        "camera_type": camera_type,
        "status": "detected"
    }

    supabase.table("detection_logs").insert(data).execute()

    return True

@app.get("/")
def home():
    return {"message": "Face Registration API is running"}

# old api read all users, file json local
# @app.get("/users")
# def get_users():
#     if not os.path.exists(USERS_FILE):
#         return {
#             "success": True,
#             "total": 0,
#             "users": []
#         }

#     with open(USERS_FILE, "r", encoding="utf-8") as f:
#         users = json.load(f)

#     return {
#         "success": True,
#         "total": len(users),
#         "users": list(users.values())
#     }

@app.get("/users")
def get_users():
    users = load_users()

    return {
        "success": True,
        "total": len(users),
        "users": list(users.values())
    }



@app.get("/users/{user_id}/embedding-status")
def embedding_status(user_id: str):
    embedding_path = f"storage/embeddings/{user_id}.npy"

    pose_files = {
        "front": os.path.exists(f"storage/embeddings/{user_id}_front.npy"),
        "left": os.path.exists(f"storage/embeddings/{user_id}_left.npy"),
        "right": os.path.exists(f"storage/embeddings/{user_id}_right.npy"),
    }

    return {
        "success": True,
        "user_id": user_id,
        "main_embedding_exists": os.path.exists(embedding_path),
        "pose_embeddings": pose_files
    }

def calculate_blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return blur_score

def calculate_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

# old function to save user profile at local json file
# def save_user_profile(user_id, full_name, role):
#     users = {}

#     if os.path.exists(USERS_FILE):
#         with open(USERS_FILE, "r", encoding="utf-8") as f:
#             users = json.load(f)

#     users[user_id] = {
#         "user_id": user_id,
#         "full_name": full_name,
#         "role": role,
#         "created_at": datetime.now().isoformat()
#     }

#     with open(USERS_FILE, "w", encoding="utf-8") as f:
#         json.dump(users, f, indent=4, ensure_ascii=False)

def save_user_profile(user_id, full_name, role):
    data = {
        "user_id": user_id,
        "full_name": full_name,
        "role": role,
        "created_at": datetime.now().isoformat()
    }

    supabase.table("users").upsert(data, on_conflict="user_id").execute()

def validate_pose(face, pose):
    landmarks = face.kps

    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]

    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_distance = abs(right_eye[0] - left_eye[0])

    if eye_distance == 0:
        return False, "Invalid face landmarks"

    nose_offset = (nose[0] - eye_center_x) / eye_distance

    if pose == "front":
        if abs(nose_offset) <= 0.15:
            return True, "Valid front pose"
        return False, "Please look straight at the camera."

    if pose == "left":
        if nose_offset > 0.10:
            return True, "Valid left pose"
        return False, "Please turn your face more to the left."

    if pose == "right":
        if nose_offset < -0.10:
            return True, "Valid right pose"
        return False, "Please turn your face more to the right."

    return True, "Pose not checked"

def validate_face_position(face, img):
    bbox = face.bbox
    x1, y1, x2, y2 = bbox

    face_center_x = (x1 + x2) / 2
    face_center_y = (y1 + y2) / 2

    img_height, img_width = img.shape[:2]

    frame_center_x = img_width / 2
    frame_center_y = img_height / 2

    offset_x = abs(face_center_x - frame_center_x) / img_width
    offset_y = abs(face_center_y - frame_center_y) / img_height

    if offset_x > 0.18:
        return False, "Please move your face to the center of the frame."

    if offset_y > 0.20:
        return False, "Please align your face vertically inside the frame."

    return True, "Face position is valid"

# Crop face by bbox 
def crop_face_by_bbox(img, face, margin_ratio=0.25, output_size=(224, 224)):
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox

    img_height, img_width = img.shape[:2]

    face_width = x2 - x1
    face_height = y2 - y1

    margin_x = int(face_width * margin_ratio)
    margin_y = int(face_height * margin_ratio)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(img_width, x2 + margin_x)
    y2 = min(img_height, y2 + margin_y)

    face_crop = img[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None

    face_crop = cv2.resize(face_crop, output_size)

    return face_crop

# upload crop face img to Supabase
def upload_file_to_supabase(bucket_name, cloud_path, local_path, content_type):
    with open(local_path, "rb") as file:
        supabase.storage.from_(bucket_name).upload(
            path=cloud_path,
            file=file,
            file_options={
                "content-type": content_type,
                "upsert": "true"
            }
        )

    return cloud_path

# Save face image metadata to Supabase
def save_face_image_metadata(user_id, pose, image_path):
    data = {
        "user_id": user_id,
        "pose": pose,
        "image_path": image_path,
        "created_at": datetime.now().isoformat()
    }
    # use upsert to avoid duplicate entries for same user and pose, if exists then update
    supabase.table("face_images").upsert(
        data,
        on_conflict="user_id,pose"
    ).execute()

@app.post("/register-face")
async def register_face(
    user_id: str = Form(...),
    full_name: str = Form(...),
    role: str = Form(...),
    pose: str = Form(...),
    image: UploadFile = File(...)
):
    image_bytes = await image.read()
    np_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    blur_score = calculate_blur_score(img)

    if blur_score < BLUR_THRESHOLD:
        return {
            "success": False,
            "message": "Image is too blurry. Please keep your face steady and try again.",
            "blur_score": blur_score
        }

    brightness = calculate_brightness(img)

    if brightness < MIN_BRIGHTNESS:
        return {
            "success": False,
            "message": "Image is too dark. Please move to a brighter place.",
            "brightness": brightness
        }

    if brightness > MAX_BRIGHTNESS:
        return {
            "success": False,
            "message": "Image is too bright. Please reduce strong light.",
            "brightness": brightness
        }

    faces = face_app.get(img)

    if len(faces) == 0:
        return {"success": False, "message": "No face detected"}

    if len(faces) > 1:
        return {"success": False, "message": "Multiple faces detected. Please make sure only one person is in front of the camera."}

    face = faces[0]
    embedding = face.embedding

    size_valid, size_message = validate_face_size(face, img)

    if not size_valid:
        return {
            "success": False,
            "message": size_message
        }

    # bbox = face.bbox
    # x1, y1, x2, y2 = bbox

    # face_width = x2 - x1
    # face_height = y2 - y1

    # img_height, img_width = img.shape[:2]

    # face_area = face_width * face_height
    # img_area = img_width * img_height

    # face_ratio = face_area / img_area
    # #Size validation
    # if face_ratio < 0.05:
    #     return {
    #         "success": False,
    #         "message": "Face is too small. Please move closer to the camera.",
    #         "face_ratio": float(face_ratio)
    #     }

    pose_valid, pose_message = validate_pose(face, pose)

    if not pose_valid:
        return {
            "success": False,
            "message": pose_message
        }

    position_valid, position_message = validate_face_position(face, img)

    if not position_valid:
        return {
            "success": False,
            "message": position_message
        }
    user_img_dir = f"storage/images/{user_id}"
    os.makedirs(user_img_dir, exist_ok=True)
    # USERS_FILE = "storage/users.json"

    # def save_user_profile(user_id, full_name, role):
    #     users = {}

    #     if os.path.exists(USERS_FILE):
    #         with open(USERS_FILE, "r", encoding="utf-8") as f:
    #             users = json.load(f)

    #     users[user_id] = {
    #         "user_id": user_id,
    #         "full_name": full_name,
    #         "role": role,
    #         "created_at": datetime.now().isoformat()
    #     }

    #     with open(USERS_FILE, "w", encoding="utf-8") as f:
    #         json.dump(users, f, indent=4, ensure_ascii=False)



    # blur_score = calculate_blur_score(img)
    
    # if blur_score < 100:  # Threshold can be adjusted
    #     return {"success": False, "message": "Image is too blurry"}


    image_path = f"{user_img_dir}/{pose}.jpg"
    embedding_path = f"storage/embeddings/{user_id}.npy"

    face_crop = crop_face_by_bbox(img, face)

    if face_crop is None:
        return {
            "success": False,
            "message": "Failed to crop face image."
        }

    cv2.imwrite(image_path, face_crop)
    # upload to Supabase
    cloud_image_path = f"{user_id}/{pose}.jpg"

    upload_file_to_supabase(
        bucket_name="face-images",
        cloud_path=cloud_image_path,
        local_path=image_path,
        content_type="image/jpeg"
    )
    # Save face image metadata to Supabase
    save_face_image_metadata(
        user_id=user_id,
        pose=pose,
        image_path=cloud_image_path
    )
        
    pose_embedding_path = f"storage/embeddings/{user_id}_{pose}.npy"
    np.save(pose_embedding_path, embedding)

    # Upload pose embedding to Supabase
    cloud_pose_embedding_path = f"{user_id}/{pose}.npy"

    upload_file_to_supabase(
        bucket_name="face-embeddings",
        cloud_path=cloud_pose_embedding_path,
        local_path=pose_embedding_path,
        content_type="application/octet-stream"
    )
    # save face embedding metadata to Supabase
    save_face_embedding_metadata(
        user_id=user_id,
        pose=pose,
        embedding_path=cloud_pose_embedding_path
    )

    all_embeddings = []

    for p in ["front", "left", "right"]:
        path = f"storage/embeddings/{user_id}_{p}.npy"
        if os.path.exists(path):
            all_embeddings.append(np.load(path))
    # average embedding from 3 poses at local
    # if len(all_embeddings) > 0:
    #     mean_embedding = np.mean(all_embeddings, axis=0)
    #     mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
    #     np.save(embedding_path, mean_embedding)

    if len(all_embeddings) > 0:
        mean_embedding = np.mean(all_embeddings, axis=0)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
        np.save(embedding_path, mean_embedding)
        # upload mean embedding to Supabase
        cloud_mean_embedding_path = f"{user_id}/mean.npy"

        upload_file_to_supabase(
            bucket_name="face-embeddings",
            cloud_path=cloud_mean_embedding_path,
            local_path=embedding_path,
            content_type="application/octet-stream"
        )

        save_face_embedding_metadata(
            user_id=user_id,
            pose="mean",
            embedding_path=cloud_mean_embedding_path
        )

    save_user_profile(user_id, full_name, role)
    return {
        "success": True,
        "message": "Face registered successfully",
        "user_id": user_id,
        "full_name": full_name,
        "role": role,
        "image_path": image_path,
        "embedding_path": embedding_path,
        "embedding_shape": embedding.shape
    }

# Test supabase connection, return one user from "users" table if exists enough to verify connection
@app.get("/test-supabase")
def test_supabase():
    response = supabase.table("users").select("*").limit(1).execute()

    return {
        "success": True,
        "message": "Connected to Supabase successfully",
        "data": response.data
    }

def validate_face_size(face, img):
    bbox = face.bbox
    x1, y1, x2, y2 = bbox

    face_width = x2 - x1
    face_height = y2 - y1

    img_height, img_width = img.shape[:2]

    face_area = face_width * face_height
    img_area = img_width * img_height

    face_ratio = face_area / img_area

    if face_ratio < 0.05:
        return False, "Face is too small. Please move closer to the camera."

    return True, "Face size is valid"

# api validate user image quality
@app.post("/validate-face")
async def validate_face_api(
    pose: str = Form(...),
    image: UploadFile = File(...)
):
    image_bytes = await image.read()
    np_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    blur_score = calculate_blur_score(img)

    if blur_score < BLUR_THRESHOLD:
        return {
            "success": False,
            "message": "Image is too blurry.",
            "blur_score": blur_score
        }

    brightness = calculate_brightness(img)

    if brightness < MIN_BRIGHTNESS:
        return {
            "success": False,
            "message": "Image is too dark.",
            "brightness": brightness
        }

    if brightness > MAX_BRIGHTNESS:
        return {
            "success": False,
            "message": "Image is too bright.",
            "brightness": brightness
        }

    faces = face_app.get(img)

    if len(faces) == 0:
        return {"success": False, "message": "No face detected."}

    if len(faces) > 1:
        return {"success": False, "message": "Multiple faces detected."}

    face = faces[0]

    size_valid, size_message = validate_face_size(face, img)
    if not size_valid:
        return {"success": False, "message": size_message}

    position_valid, position_message = validate_face_position(face, img)
    if not position_valid:
        return {"success": False, "message": position_message}

    pose_valid, pose_message = validate_pose(face, pose)
    if not pose_valid:
        return {"success": False, "message": pose_message}

    return {
        "success": True,
        "message": "Face is valid for capture."
    }



@app.get("/users-page")
def users_page():
    return FileResponse("static/users.html")

# api delete user
# @app.delete("/users/{user_id}")
# def delete_user(user_id: str):
    users = {}

    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            users = json.load(f)

    if user_id not in users:
        return {
            "success": False,
            "message": "User not found"
        }

    del users[user_id]

    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4, ensure_ascii=False)

    image_folder = f"storage/images/{user_id}"
    if os.path.exists(image_folder):
        shutil.rmtree(image_folder)

    for file in os.listdir("storage/embeddings"):
        if file.startswith(user_id):
            os.remove(f"storage/embeddings/{file}")

    return {
        "success": True,
        "message": "User deleted successfully",
        "user_id": user_id
    }

@app.delete("/users/{user_id}")
def delete_user(user_id: str):
    # Check user exists in Supabase
    user_response = (
        supabase
        .table("users")
        .select("*")
        .eq("user_id", user_id)
        .execute()
    )

    if len(user_response.data) == 0:
        return {
            "success": False,
            "message": "User not found in cloud database",
            "user_id": user_id
        }

    # Get face image metadata
    image_response = (
        supabase
        .table("face_images")
        .select("*")
        .eq("user_id", user_id)
        .execute()
    )

    #  Delete face image files from Supabase Storage
    for item in image_response.data:
        try:
            supabase.storage.from_("face-images").remove([
                item["image_path"]
            ])
        except Exception as e:
            print(f"Failed to delete image from storage: {e}")

    #  Get embedding metadata
    embedding_response = (
        supabase
        .table("face_embeddings")
        .select("*")
        .eq("user_id", user_id)
        .execute()
    )

    # Delete embedding files from Supabase Storage
    for item in embedding_response.data:
        try:
            supabase.storage.from_("face-embeddings").remove([
                item["embedding_path"]
            ])
        except Exception as e:
            print(f"Failed to delete embedding from storage: {e}")

    #  Delete metadata from cloud tables
    supabase.table("face_images").delete().eq("user_id", user_id).execute()
    supabase.table("face_embeddings").delete().eq("user_id", user_id).execute()
    supabase.table("users").delete().eq("user_id", user_id).execute()

    # Delete local cached images
    image_folder = f"storage/images/{user_id}"

    if os.path.exists(image_folder):
        shutil.rmtree(image_folder)

    # Delete local cached embeddings
    if os.path.exists("storage/embeddings"):
        for file in os.listdir("storage/embeddings"):
            if file.startswith(user_id):
                file_path = os.path.join("storage/embeddings", file)
                os.remove(file_path)

    return {
        "success": True,
        "message": "User and related cloud data deleted successfully",
        "user_id": user_id
    }



# Save face embedding metadata to Supabase
def save_face_embedding_metadata(user_id, pose, embedding_path):
    data = {
        "user_id": user_id,
        "pose": pose,
        "embedding_path": embedding_path,
        "created_at": datetime.now().isoformat()
    }
    # use upsert to avoid duplicate entries for same user and pose, will update
    supabase.table("face_embeddings").upsert(
        data,
        on_conflict="user_id,pose"
    ).execute()

# old Api check if user exists by read json local file
# @app.get("/users/{user_id}/exists")
# def check_user_exists(user_id: str):
#     if not os.path.exists(USERS_FILE):
#         return {
#             "success": True,
#             "exists": False
#         }

#     with open(USERS_FILE, "r", encoding="utf-8") as f:
#         users = json.load(f)

#     return {
#         "success": True,
#         "exists": user_id in users
#     }

# Api check if user exists by query Supabase
@app.get("/users/{user_id}/exists")
def check_user_exists(user_id: str):
    response = (
        supabase
        .table("users")
        .select("user_id")
        .eq("user_id", user_id)
        .execute()
    )

    return {
        "success": True,
        "exists": len(response.data) > 0
    }
# old function to save unknown face at local folder
# def save_unknown_face(img, confidence, camera_location):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"unknown_{timestamp}_{confidence:.3f}.jpg"
    path = f"storage/unknown_faces/{filename}"

    cv2.imwrite(path, img)

    return path

# Save attendance candidate
# def save_attendance_candidate(user_id, full_name, role, confidence, camera_id, camera_location, camera_type):
    candidates = []

    if os.path.exists(ATTENDANCE_CANDIDATE_FILE):
        with open(ATTENDANCE_CANDIDATE_FILE, "r", encoding="utf-8") as f:
            candidates = json.load(f)

    candidate = {
        "user_id": user_id,
        "full_name": full_name,
        "role": role,
        "timestamp": datetime.now().isoformat(),
        "confidence": confidence,
        "camera_id": camera_id,
        "camera_location": camera_location,
        "camera_type": camera_type,
        "status": "attendance_candidate",
        "sent_to_school_system": True
    }

    candidates.append(candidate)

    with open(ATTENDANCE_CANDIDATE_FILE, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=4, ensure_ascii=False)

    return candidate

def save_unknown_face(img, confidence, camera_location):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"unknown_{timestamp}_{confidence:.3f}.jpg"

    local_path = f"storage/unknown_faces/{filename}"
    cloud_path = filename

    cv2.imwrite(local_path, img)

    upload_file_to_supabase(
        bucket_name="unknown-faces",
        cloud_path=cloud_path,
        local_path=local_path,
        content_type="image/jpeg"
    )

    data = {
        "filename": filename,
        "image_path": cloud_path,
        "confidence": confidence,
        "camera_location": camera_location,
        "timestamp": datetime.now().isoformat()
    }

    supabase.table("unknown_faces").insert(data).execute()

    return cloud_path

def save_attendance_candidate(user_id, full_name, role, confidence, camera_id, camera_location, camera_type):
    candidate = {
        "user_id": user_id,
        "full_name": full_name,
        "role": role,
        "timestamp": datetime.now().isoformat(),
        "confidence": confidence,
        "camera_id": camera_id,
        "camera_location": camera_location,
        "camera_type": camera_type,
        "status": "attendance_candidate",
        "sent_to_school_system": True
    }

    supabase.table("attendance_candidates").insert(candidate).execute()

    return candidate

# Load embedding from Supabase
def load_embedding_from_cloud(embedding_path):
    file_bytes = supabase.storage.from_("face-embeddings").download(embedding_path)

    embedding = np.load(BytesIO(file_bytes))

    return embedding

# Load mean embeddings for all users from Supabase, return dict of user_id to embedding
def load_cloud_mean_embeddings():
    response = (
        supabase
        .table("face_embeddings")
        .select("user_id, pose, embedding_path")
        .eq("pose", "mean")
        .execute()
    )

    embeddings = {}

    for item in response.data:
        user_id = item["user_id"]
        embedding_path = item["embedding_path"]

        try:
            embeddings[user_id] = load_embedding_from_cloud(embedding_path)
        except Exception as e:
            print(f"Failed to load embedding for {user_id}: {e}")

    return embeddings

# new function to recognize faces from frame
def recognize_faces_from_frame(img):
    faces = face_app.get(img)

    if len(faces) == 0:
        return {
            "success": False,
            "message": "No face detected"
        }



    users = load_users()

    # put cloud embedding loading here to avoid loading if no faces detected, but only load once for all faces in this image, not load for each face to save time
    cloud_embeddings = load_cloud_mean_embeddings()
    if len(users) == 0:
        return {
            "success": False,
            "message": "No registered users found"
        }

    results = []

    for face in faces:
        current_embedding = face.embedding

        bbox = face.bbox
        x1, y1, x2, y2 = bbox.astype(int)

        bbox_data = {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2)
        }

        best_match = None
        best_score = -1

        # cloud_embeddings = load_cloud_mean_embeddings()

        for user_id, user_info in users.items():
            if user_id not in cloud_embeddings:
                continue

            saved_embedding = cloud_embeddings[user_id]
            score = cosine_similarity(current_embedding, saved_embedding)

            if score > best_score:
                best_score = score
                best_match = user_info

        if best_match is None:
            results.append({
                "recognized": False,
                "message": "No registered embeddings found",
                "confidence": best_score,
                "bbox": bbox_data
            })
            continue

        if best_score < RECOGNITION_THRESHOLD:
            results.append({
                "recognized": False,
                "message": "Unknown person",
                "confidence": best_score,
                "bbox": bbox_data
            })
            continue


        results.append({
            "recognized": True,
            "message": "User recognized",
            "user_id": best_match["user_id"],
            "full_name": best_match["full_name"],
            "role": best_match["role"],
            "confidence": best_score,
            "bbox": bbox_data
        })
    return results

# api recognize face for attendance
@app.post("/recognize-face")
async def recognize_face(
    image: UploadFile = File(...),
    camera_id: str = Form("unknown_camera"),
    camera_location: str = Form("Unknown"),
    camera_type: str = Form("unknown")
):
    image_bytes = await image.read()
    np_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    results = recognize_faces_from_frame(img)

    if len(results) == 0:
        return {
            "success": False,
            "message": "No face detected"
        }

    final_results = []

    for face in results:
        # Case 1: recognized user
        if face.get("recognized"):
            logged = save_attendance_log(
                face["user_id"],
                face["full_name"],
                face["role"],
                face["confidence"],
                camera_id,
                camera_location,
                camera_type
            )

            attendance_candidate = None

            if face["role"] == "student":
                attendance_candidate = save_attendance_candidate(
                    face["user_id"],
                    face["full_name"],
                    face["role"],
                    face["confidence"],
                    camera_id,
                    camera_location,
                    camera_type
                )

            final_results.append({
                "recognized": True,
                "message": "User recognized",
                "user_id": face["user_id"],
                "full_name": face["full_name"],
                "role": face["role"],
                "confidence": face["confidence"],
                "status": "present",
                "logged": logged,
                "camera_id": camera_id,
                "camera_location": camera_location,
                "camera_type": camera_type,
                "bbox": face.get("bbox"),
                "attendance_candidate": attendance_candidate
            })

        # Case 2: unknown / not recognized
        else:
            unknown_path = save_unknown_face(
                img,
                face.get("confidence", -1),
                camera_location
            )

            final_results.append({
                "recognized": False,
                "message": face.get("message", "Unknown person"),
                "confidence": face.get("confidence", -1),
                "camera_id": camera_id,
                "camera_location": camera_location,
                "camera_type": camera_type,
                "unknown_image_path": unknown_path,
                "bbox": face.get("bbox")
            })

    return {
        "success": True,
        "total_faces": len(final_results),
        "results": final_results
    }

# api view unknown faces
@app.get("/unknown-faces")
def get_unknown_faces():
    response = (
        supabase
        .table("unknown_faces")
        .select("*")
        .order("timestamp", desc=True)
        .execute()
    )

    files = []

    for item in response.data:
        public_url = supabase.storage.from_("unknown-faces").get_public_url(item["image_path"])

        files.append({
            "filename": item["filename"],
            "path": item["image_path"],
            "public_url": public_url,
            "confidence": item["confidence"],
            "camera_location": item["camera_location"],
            "timestamp": item["timestamp"]
        })

    return {
        "success": True,
        "total": len(files),
        "files": files
    }
    folder = "storage/unknown_faces"

    if not os.path.exists(folder):
        return {
            "success": True,
            "total": 0,
            "files": []
        }

    files = []

    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            files.append({
                "filename": file,
                "path": f"{folder}/{file}"
            })

    return {
        "success": True,
        "total": len(files),
        "files": files
    }


# old api view attendance logs from local json file
# @app.get("/attendance-logs")
# def get_attendance_logs():
    if not os.path.exists(ATTENDANCE_FILE):
        return {
            "success": True,
            "total": 0,
            "logs": []
        }

    with open(ATTENDANCE_FILE, "r", encoding="utf-8") as f:
        logs = json.load(f)

    return {
        "success": True,
        "total": len(logs),
        "logs": logs
    }

@app.get("/attendance-logs")
def get_attendance_logs():
    response = (
        supabase
        .table("detection_logs")
        .select("*")
        .order("timestamp", desc=True)
        .execute()
    )

    return {
        "success": True,
        "total": len(response.data),
        "logs": response.data
    }

# api attendance page
@app.get("/attendance-page")
def attendance_page():
    return FileResponse("static/attendance.html")

# api dashboard page
@app.get("/dashboard-summary")
def dashboard_summary():
    users = load_users()

    today = datetime.now().date().isoformat()

    logs_response = (
        supabase
        .table("detection_logs")
        .select("*")
        .gte("timestamp", today + "T00:00:00")
        .lte("timestamp", today + "T23:59:59")
        .execute()
    )

    unknown_response = (
        supabase
        .table("unknown_faces")
        .select("*")
        .execute()
    )

    unique_present_users = set(
        log["user_id"] for log in logs_response.data
        if log.get("user_id")
    )

    return {
        "success": True,
        "date": today,
        "registered_users_count": len(users),
        "today_total_present": len(unique_present_users),
        "unknown_detections": len(unknown_response.data),
        "latest_attendance_logs": logs_response.data[-5:][::-1]
    }
# export attendance logs as CSV
@app.get("/attendance-logs/export")
def export_attendance_logs():
    export_path = "storage/attendance_logs.csv"

    # read logs local json file
    # logs = []

    # if os.path.exists(ATTENDANCE_FILE):
    #     with open(ATTENDANCE_FILE, "r", encoding="utf-8") as f:
    #         logs = json.load(f)

    response = (
        supabase
        .table("detection_logs")
        .select("*")
        .order("timestamp", desc=True)
        .execute()
    )

    logs = response.data
    with open(export_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "user_id",
            "full_name",
            "role",
            "timestamp",
            "camera_location",
            "confidence",
            "status"
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for log in logs:
            writer.writerow({
                "user_id": log.get("user_id", ""),
                "full_name": log.get("full_name", ""),
                "role": log.get("role", ""),
                "timestamp": log.get("timestamp", ""),
                "camera_location": log.get("camera_location", ""),
                "confidence": log.get("confidence", ""),
                "status": log.get("status", "")
            })

    return FileResponse(
        export_path,
        media_type="text/csv",
        filename="attendance_logs.csv"
    )

# api clear attendance logs
@app.delete("/attendance-logs")
def clear_attendance_logs():
    # fiter by id greater than 0 to avoid deleting any potential system metadata if exists with id 0, but in general should be safe to delete all logs
    supabase.table("detection_logs").delete().gt("id", 0).execute()

    return {
        "success": True,
        "message": "Detection logs cleared successfully."
    }

# api clear unknown faces
@app.delete("/unknown-faces")
def clear_unknown_faces():
    response = supabase.table("unknown_faces").select("*").execute()

    for item in response.data:
        try:
            supabase.storage.from_("unknown-faces").remove([item["image_path"]])
        except Exception as e:
            print(f"Failed to delete unknown face file: {e}")

    supabase.table("unknown_faces").delete().gt("id", 0).execute()

    return {
        "success": True,
        "message": "Unknown faces cleared successfully"
    }

# API view attendance candidates
@app.get("/attendance-candidates")
def get_attendance_candidates():
    response = (
        supabase
        .table("attendance_candidates")
        .select("*")
        .order("timestamp", desc=True)
        .execute()
    )

    return {
        "success": True,
        "total": len(response.data),
        "candidates": response.data
    }

# api clear attendance logs
@app.delete("/attendance-candidates")
def clear_attendance_candidates():
    supabase.table("attendance_candidates").delete().gt("id", 0).execute()

    return {
        "success": True,
        "message": "Attendance candidates cleared successfully."
    }

# api check cloud health
@app.get("/cloud-health")
def cloud_health():
    health = {
        "success": True,
        "database": {},
        "storage": {}
    }

    tables = [
        "users",
        "face_images",
        "face_embeddings",
        "detection_logs",
        "attendance_candidates",
        "unknown_faces"
    ]

    for table in tables:
        try:
            response = (
                supabase
                .table(table)
                .select("*")
                .limit(1)
                .execute()
            )

            health["database"][table] = {
                "status": "ok",
                "sample_count": len(response.data)
            }

        except Exception as e:
            health["success"] = False
            health["database"][table] = {
                "status": "error",
                "message": str(e)
            }

    buckets = [
        "face-images",
        "face-embeddings",
        "unknown-faces",
        "exports"
    ]

    for bucket in buckets:
        try:
            response = supabase.storage.from_(bucket).list()

            health["storage"][bucket] = {
                "status": "ok",
                "items_count": len(response)
            }

        except Exception as e:
            health["success"] = False
            health["storage"][bucket] = {
                "status": "error",
                "message": str(e)
            }

    return health


# ------------------------------------------------
# api create video session
@app.post("/video-sessions")
def create_video_session(
    session_id: str = Form(...),
    class_name: str = Form(...),
    camera_id: str = Form(...),
    camera_location: str = Form(...),
    video_name: str = Form(...),
    duration_seconds: float = Form(0),
    total_windows: int = Form(0),
    sample_interval_seconds: float = Form(5)
):
    data = {
        "session_id": session_id,
        "class_name": class_name,
        "camera_id": camera_id,
        "camera_location": camera_location,
        "video_name": video_name,
        "duration_seconds": duration_seconds,
        "total_windows": total_windows,
        "sample_interval_seconds": sample_interval_seconds
    }

    supabase.table("video_sessions").upsert(
        data,
        on_conflict="session_id"
    ).execute()

    return {
        "success": True,
        "message": "Video session created successfully",
        "session": data
    }

# api view video sessions
@app.get("/video-sessions")
def get_video_sessions():
    response = (
        supabase
        .table("video_sessions")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )

    return {
        "success": True,
        "total": len(response.data),
        "sessions": response.data
    }

# api process video 
@app.post("/process-video")
async def process_video(
    session_id: str = Form(...),
    class_name: str = Form(...),
    camera_id: str = Form(...),
    camera_location: str = Form(...),
    sample_interval_seconds: float = Form(5),
    video: UploadFile = File(...)
):
    video_path = f"storage/videos/{session_id}_{video.filename}"

    with open(video_path, "wb") as f:
        f.write(await video.read())

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {
            "success": False,
            "message": "Failed to open video file."
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if fps == 0:
        cap.release()
        return {
            "success": False,
            "message": "Invalid video FPS."
        }

    duration_seconds = frame_count / fps
    frame_step = int(fps * sample_interval_seconds)

    if frame_step <= 0:
        frame_step = 1

    total_windows = 0
    frame_index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_index % frame_step == 0:

            total_windows += 1

            frame_time_seconds = frame_index / fps

            results = recognize_faces_from_frame(frame)
# add video evidence frame if any recognized faces detected
            if len(results) > 0:
                save_video_evidence_frame(
                    session_id=session_id,
                    class_name=class_name,
                    camera_id=camera_id,
                    frame_time_seconds=frame_time_seconds,
                    frame=frame,
                    results=results
                )

            for face in results:
                log_data = {
                    "session_id": session_id,
                    "class_name": class_name,
                    "camera_id": camera_id,
                    "camera_location": camera_location,
                    "frame_time_seconds": frame_time_seconds,
                    "user_id": face.get("user_id"),
                    "full_name": face.get("full_name"),
                    "confidence": face.get("confidence"),
                    "status": (
                        "recognized"
                        if face.get("recognized")
                        else "unknown"
                    )
                }

                supabase.table(
                    "video_detection_logs"
                ).insert(log_data).execute()

        frame_index += 1

    cap.release()

    session_data = {
        "session_id": session_id,
        "class_name": class_name,
        "camera_id": camera_id,
        "camera_location": camera_location,
        "video_name": video.filename,
        "duration_seconds": duration_seconds,
        "total_windows": total_windows,
        "sample_interval_seconds": sample_interval_seconds
    }

    supabase.table("video_sessions").upsert(
        session_data,
        on_conflict="session_id"
    ).execute()
    summary_result = generate_video_attendance_summary(session_id)
    return {
        "success": True,
        "message": "Video processed successfully.",
        "session_id": session_id,
        "duration_seconds": duration_seconds,
        "total_windows": total_windows,
        "summary_result": summary_result
    }

def generate_video_attendance_summary(session_id):
    session_response = (
        supabase
        .table("video_sessions")
        .select("*")
        .eq("session_id", session_id)
        .limit(1)
        .execute()
    )

    if len(session_response.data) == 0:
        return {
            "success": False,
            "message": "Video session not found"
        }

    session = session_response.data[0]
    total_windows = session.get("total_windows", 0)
    class_name = session.get("class_name")

    if total_windows == 0:
        return {
            "success": False,
            "message": "Total windows is 0"
        }

    logs_response = (
        supabase
        .table("video_detection_logs")
        .select("*")
        .eq("session_id", session_id)
        .eq("status", "recognized")
        .execute()
    )

    logs = logs_response.data

    student_stats = {}

    for log in logs:
        user_id = log.get("user_id")

        if not user_id:
            continue

        if user_id not in student_stats:
            student_stats[user_id] = {
                "user_id": user_id,
                "full_name": log.get("full_name"),
                "detected_windows": set(),
                "last_seen_seconds": 0
            }

        frame_time = log.get("frame_time_seconds")

        student_stats[user_id]["detected_windows"].add(frame_time)

        if frame_time > student_stats[user_id]["last_seen_seconds"]:
            student_stats[user_id]["last_seen_seconds"] = frame_time

    summaries = []

    for user_id, stat in student_stats.items():
        detected_windows = len(stat["detected_windows"])
        presence_ratio = detected_windows / total_windows

        suggested_status = "present" if presence_ratio >= 0.5 else "absent"

        leave_status = "stayed"

        if stat["last_seen_seconds"] < session["duration_seconds"] - 60:
            leave_status = "possible_left_classroom"

        summary = {
            "session_id": session_id,
            "class_name": class_name,
            "user_id": user_id,
            "full_name": stat["full_name"],
            "detected_windows": detected_windows,
            "total_windows": total_windows,
            "presence_ratio": presence_ratio,
            "suggested_status": suggested_status,
            "last_seen_seconds": stat["last_seen_seconds"],
            "leave_status": leave_status
        }

        summaries.append(summary)

    supabase.table("video_attendance_summary").delete().eq(
        "session_id",
        session_id
    ).execute()

    if len(summaries) > 0:
        supabase.table("video_attendance_summary").insert(summaries).execute()

    return {
        "success": True,
        "session_id": session_id,
        "total_students": len(summaries),
        "summaries": summaries
    }


@app.post("/video-summary/{session_id}/generate")
def generate_video_summary_api(session_id: str):
    return generate_video_attendance_summary(session_id)

@app.get("/video-summary/{session_id}")
def get_video_summary(session_id: str):

    response = (
        supabase
        .table("video_attendance_summary")
        .select("*")
        .eq("session_id", session_id)
        .order("presence_ratio", desc=True)
        .execute()
    )

    return {
        "success": True,
        "session_id": session_id,
        "total": len(response.data),
        "summaries": response.data
    }

@app.get("/video-analysis-page")
def video_analysis_page():
    return FileResponse("static/video_analysis.html")
@app.get("/video-summary-page")
def video_summary_page():
    return FileResponse("static/video_summary.html")
@app.get("/video-evidence-page")
def video_evidence_page():
    return FileResponse("static/video_evidence.html")

# Draw bounding boxes and labels on video frames for evidence
def draw_video_evidence_frame(frame, results):
    annotated = frame.copy()

    for face in results:
        bbox = face.get("bbox")

        if not bbox:
            continue

        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        if face.get("recognized"):
            label = f'{face.get("full_name")} {face.get("confidence", 0):.2f}'
            color = (0, 255, 0)
        else:
            label = f'Unknown {face.get("confidence", 0):.2f}'
            color = (0, 0, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return annotated

def save_video_evidence_frame(session_id, class_name, camera_id, frame_time_seconds, frame, results):
    os.makedirs("storage/video_evidence_frames", exist_ok=True)

    filename = f"{session_id}_{int(frame_time_seconds)}.jpg"
    local_path = f"storage/video_evidence_frames/{filename}"
    cloud_path = f"{session_id}/{filename}"

    annotated_frame = draw_video_evidence_frame(frame, results)

    cv2.imwrite(local_path, annotated_frame)

    upload_file_to_supabase(
        bucket_name="video-evidence-frames",
        cloud_path=cloud_path,
        local_path=local_path,
        content_type="image/jpeg"
    )

    data = {
        "session_id": session_id,
        "class_name": class_name,
        "camera_id": camera_id,
        "frame_time_seconds": frame_time_seconds,
        "image_path": cloud_path,
        "detected_count": len(results)
    }

    supabase.table("video_evidence_frames").insert(data).execute()

    return cloud_path

@app.get("/video-evidence/{session_id}")
def get_video_evidence_frames(session_id: str):
    response = (
        supabase
        .table("video_evidence_frames")
        .select("*")
        .eq("session_id", session_id)
        .order("frame_time_seconds")
        .execute()
    )

    frames = []

    for item in response.data:
        public_url = supabase.storage.from_("video-evidence-frames").get_public_url(
            item["image_path"]
        )

        frames.append({
            "id": item["id"],
            "session_id": item["session_id"],
            "frame_time_seconds": item["frame_time_seconds"],
            "detected_count": item["detected_count"],
            "image_path": item["image_path"],
            "public_url": public_url
        })

    return {
        "success": True,
        "session_id": session_id,
        "total": len(frames),
        "frames": frames
    }

@app.get("/video-sessions")
def get_video_sessions():

    response = (
        supabase
        .table("video_sessions")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )

    return {
        "success": True,
        "sessions": response.data
    }