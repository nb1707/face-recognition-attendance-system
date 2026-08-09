import os
import sys
from io import BytesIO

import cv2
import numpy as np
from dotenv import load_dotenv
from supabase import create_client
import argparse

# allow importing from backend directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


from recognition.model_factory import get_model


load_dotenv(os.path.join(BACKEND_DIR, ".env"))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_URL or SUPABASE_KEY is missing from .env"
    )

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


FACE_IMAGE_BUCKET = "face-images"
FACE_EMBEDDING_BUCKET = "face-embeddings"

POSES = ["front", "left", "right"]


def download_image_from_cloud(
    image_path: str
) -> np.ndarray:
    file_bytes = (
        supabase.storage
        .from_(FACE_IMAGE_BUCKET)
        .download(image_path)
    )

    np_buffer = np.frombuffer(
        file_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        np_buffer,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            f"Could not decode image: {image_path}"
        )

    return image


def upload_numpy_embedding(
    cloud_path: str,
    embedding: np.ndarray
):
    memory_file = BytesIO()

    np.save(
        memory_file,
        embedding.astype(np.float32)
    )

    memory_file.seek(0)

    supabase.storage.from_(
        FACE_EMBEDDING_BUCKET
    ).upload(
        path=cloud_path,
        file=memory_file.getvalue(),
        file_options={
            "content-type": "application/octet-stream",
            "upsert": "true"
        }
    )


def save_embedding_metadata(
    user_id: str,
    pose: str,
    model_name: str,
    embedding_path: str
):
    data = {
        "user_id": user_id,
        "pose": pose,
        "model_name": model_name,
        "embedding_path": embedding_path
    }

    supabase.table(
        "face_embeddings"
    ).upsert(
        data,
        on_conflict="user_id,pose,model_name"
    ).execute()


def load_registered_users():
    response = (
        supabase
        .table("users")
        .select("user_id,full_name,role")
        .execute()
    )

    return response.data


def load_face_image_metadata(
    user_id: str
):
    response = (
        supabase
        .table("face_images")
        .select(
            "pose,image_path,aligned_image_path"
        )
        .eq("user_id", user_id)
        .execute()
    )

    return {
        item["pose"]: {
            "image_path": item.get(
                "image_path"
            ),
            "aligned_image_path": item.get(
                "aligned_image_path"
            )
        }
        for item in response.data
    }

def generate_user_embeddings(
    user,
    model
):
    user_id = user["user_id"]

    print(
        f"\nProcessing user: "
        f"{user_id} - {user.get('full_name')}"
    )

    image_metadata = load_face_image_metadata(
        user_id
    )

    pose_embeddings = []

    for pose in POSES:
        pose_metadata = image_metadata.get(
            pose
        )

        if not pose_metadata:
            print(
                f"  SKIPPED {pose}: "
                "image metadata missing"
            )
            continue

        aligned_image_path = (
            pose_metadata.get(
                "aligned_image_path"
            )
        )

        if not aligned_image_path:
            print(
                f"  FAILED {pose}: "
                "aligned image path missing"
            )
            continue

        try:
            aligned_face = download_image_from_cloud(
                aligned_image_path
            )

            if (
                aligned_face is None
                or aligned_face.size == 0
            ):
                print(
                    f"  FAILED {pose}: "
                    "could not decode aligned image"
                )
                continue

            if aligned_face.shape[:2] != (112, 112):
                aligned_face = cv2.resize(
                    aligned_face,
                    (112, 112),
                    interpolation=cv2.INTER_LINEAR
                )

            embedding = model.extract_embedding(
                aligned_face
            )

            embedding = np.asarray(
                embedding,
                dtype=np.float32
            ).reshape(-1)

            if embedding.size == 0:
                print(
                    f"  FAILED {pose}: "
                    "empty embedding"
                )
                continue

            cloud_path = (
                f"{user_id}/"
                f"{model.model_name}/"
                f"{pose}.npy"
            )

            upload_numpy_embedding(
                cloud_path=cloud_path,
                embedding=embedding
            )

            save_embedding_metadata(
                user_id=user_id,
                pose=pose,
                model_name=model.model_name,
                embedding_path=cloud_path
            )

            pose_embeddings.append(
                embedding
            )

            print(
                f"  OK {pose}: "
                f"shape={embedding.shape}"
            )

        except Exception as error:
            print(
                f"  FAILED {pose}: "
                f"{error}"
            )

    if len(pose_embeddings) != 3:
        print(
            "  Mean embedding not generated: "
            f"only {len(pose_embeddings)}/3 "
            "poses succeeded"
        )

        return False

    embedding_shapes = {
        embedding.shape
        for embedding in pose_embeddings
    }

    if len(embedding_shapes) != 1:
        print(
            "  Mean embedding failed: "
            f"inconsistent embedding shapes "
            f"{embedding_shapes}"
        )

        return False

    mean_embedding = np.mean(
        pose_embeddings,
        axis=0
    ).astype(np.float32)

    mean_norm = np.linalg.norm(
        mean_embedding
    )

    if mean_norm == 0:
        print(
            "  Mean embedding failed: "
            "zero norm"
        )

        return False

    mean_embedding = (
        mean_embedding / mean_norm
    ).astype(np.float32)

    mean_cloud_path = (
        f"{user_id}/"
        f"{model.model_name}/"
        "mean.npy"
    )

    try:
        upload_numpy_embedding(
            cloud_path=mean_cloud_path,
            embedding=mean_embedding
        )

        save_embedding_metadata(
            user_id=user_id,
            pose="mean",
            model_name=model.model_name,
            embedding_path=mean_cloud_path
        )

        print(
            f"  OK mean: "
            f"shape={mean_embedding.shape}"
        )

        return True

    except Exception as error:
        print(
            f"  FAILED mean: {error}"
        )

        return False


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Generate model-specific face embeddings "
            "from existing registration images."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "buffalo_l",
            "facenet",
            "adaface",
            "arcface_r100"
        ],
        help="Embedding model name."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    model_name = args.model

    model = get_model(
        model_name
    )

    users = load_registered_users()

    print(
        f"Generating {model_name} embeddings "
        f"for {len(users)} users"
    )

    success_count = 0
    failed_count = 0

    for user in users:
        success = generate_user_embeddings(
            user=user,
            model=model
        )

        if success:
            success_count += 1
        else:
            failed_count += 1

    print("\nGeneration completed")
    print(f"Model: {model_name}")
    print(f"Successful users: {success_count}")
    print(f"Failed users: {failed_count}")


if __name__ == "__main__":
    main()