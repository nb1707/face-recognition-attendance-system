from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from enrollment.errors import EnrollmentError


class StartEnrollmentRequest(BaseModel):
    user_id: str
    full_name: str
    role: str
    allow_reenroll: bool = False


def create_enrollment_router(service):
    router = APIRouter(prefix="/enrollments", tags=["local-enrollment"])

    def error_response(error):
        payload = {
            "success": False, "accepted": False, "error_code": error.code,
            "message": error.message, "sample_id": None,
            "quality_decision": "REJECTED", "pose_accepted": False,
            "duplicate_status": error.code if error.code in {"EXACT_DUPLICATE", "NEAR_DUPLICATE"} else "NOT_EVALUATED",
            **error.details,
        }
        return JSONResponse(status_code=error.status_code, content=payload)

    @router.post("", status_code=201)
    def start_enrollment(request: StartEnrollmentRequest):
        try:
            return service.start_session(request.user_id, request.full_name, request.role,
                                         request.allow_reenroll)
        except EnrollmentError as error:
            return error_response(error)

    @router.get("/{session_id}")
    def inspect_enrollment(session_id: str):
        try:
            return service.get_session(session_id)
        except EnrollmentError as error:
            return error_response(error)

    @router.post("/{session_id}/samples", status_code=201)
    async def submit_sample(session_id: str, pose: str = Form(...), image: UploadFile = File(...)):
        try:
            return service.submit_sample(session_id, pose, await image.read())
        except EnrollmentError as error:
            return error_response(error)

    @router.delete("/{session_id}/samples/{sample_id}")
    def delete_sample(session_id: str, sample_id: str):
        try:
            return service.delete_sample(session_id, sample_id)
        except EnrollmentError as error:
            return error_response(error)

    @router.get("/{session_id}/samples/{sample_id}/preview")
    def sample_preview(session_id: str, sample_id: str):
        try:
            return FileResponse(service.preview_path(session_id, sample_id), media_type="image/jpeg",
                                headers={"Cache-Control": "no-store"})
        except EnrollmentError as error:
            return error_response(error)

    @router.post("/{session_id}/finalize")
    def finalize_enrollment(session_id: str):
        try:
            return service.finalize(session_id)
        except EnrollmentError as error:
            return error_response(error)

    @router.delete("/{session_id}")
    def cancel_enrollment(session_id: str):
        try:
            service.cancel_session(session_id)
            return {"success": True, "message": "Unfinished enrollment session cancelled."}
        except EnrollmentError as error:
            return error_response(error)

    return router
