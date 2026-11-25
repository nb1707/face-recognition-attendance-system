class EnrollmentError(Exception):
    def __init__(self, code, message, status_code=400, details=None):
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.status_code = int(status_code)
        self.details = dict(details or {})


MESSAGES = {
    "NO_FACE": "No face detected. Move closer and face the camera.",
    "MULTIPLE_FACES": "Only one person should be visible.",
    "FACE_TOO_SMALL": "Move closer to the camera.",
    "BLURRED_FACE": "Hold still and try again.",
    "POOR_LIGHTING": "Improve the lighting and try again.",
    "LOW_DETECTOR_CONFIDENCE": "Make your face clearer and fully visible.",
    "INVALID_LANDMARKS": "Face landmarks could not be measured reliably.",
    "WRONG_POSE": "Adjust your face to match the requested pose.",
    "FACE_NOT_CENTERED": "Center your face inside the guide.",
    "EXACT_DUPLICATE": "This is the same frame as an accepted sample.",
    "NEAR_DUPLICATE": "Sample too similar to a previous sample; vary your expression slightly.",
    "POSE_QUOTA_REACHED": "The requested pose already has enough samples.",
    "INVALID_SESSION": "Enrollment session was not found.",
    "SESSION_FINALIZED": "This enrollment session is already complete.",
    "SESSION_NOT_READY": "Enrollment does not yet have all required samples.",
    "INVALID_IMAGE": "The uploaded image is invalid or unsupported.",
    "STORAGE_ERROR": "Local enrollment storage failed.",
    "EMBEDDING_ERROR": "AdaFace embedding generation failed.",
    "GALLERY_PUBLISH_ERROR": "Local gallery publication failed; accepted samples were retained.",
    "USER_ALREADY_ENROLLED": "This user already has a completed local enrollment.",
}


def enrollment_error(code, *, status_code=400, details=None, message=None):
    return EnrollmentError(code, message or MESSAGES.get(code, code), status_code, details)
