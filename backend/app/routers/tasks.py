from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import shutil

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..auth.database import get_db_connection

from ..security import get_current_user

BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_ROOT = BASE_DIR / "uploads"
TASK_UPLOAD_DIR = UPLOAD_ROOT / "tasks"
TASK_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/tasks", tags=["tasks"])


def _iso_or_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _save_upload(upload: UploadFile) -> tuple[str, str, str]:
    original_name = upload.filename or "upload"
    suffix = Path(original_name).suffix
    unique_name = f"{uuid4().hex}{suffix}"
    destination = TASK_UPLOAD_DIR / unique_name
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return str(destination), f"/uploads/tasks/{unique_name}", original_name


class StudentInfo(BaseModel):
    id: int
    username: str


class TaskSummary(BaseModel):
    id: int
    title: str
    description: Optional[str]
    teacher_id: int
    student_ids: List[int]
    deadline: Optional[str]
    file_url: Optional[str]
    original_filename: Optional[str]
    created_at: str
    student_status: Optional[str] = None
    submitted_at: Optional[str] = None
    is_overdue: Optional[bool] = None
    submitted_count: Optional[int] = None
    total_assignees: Optional[int] = None


class TaskDetail(TaskSummary):
    teacher_name: Optional[str]
    students: List[StudentInfo]


class SubmissionResponse(BaseModel):
    id: Optional[int]
    student_id: int
    student_name: str
    file_url: Optional[str]
    status: str
    submitted_at: Optional[str]


def _fetch_students(conn, student_ids: List[int]) -> List[StudentInfo]:
    if not student_ids:
        return []
    placeholders = ",".join(["?"] * len(student_ids))
    rows = conn.execute(
        f"SELECT id, username FROM users WHERE id IN ({placeholders})",
        tuple(student_ids),
    ).fetchall()
    return [StudentInfo(id=row["id"], username=row["username"]) for row in rows]


def _get_assignees(conn, task_id: int) -> List[int]:
    rows = conn.execute(
        "SELECT student_id FROM task_assignees WHERE task_id = ?",
        (task_id,),
    ).fetchall()
    return [row["student_id"] for row in rows]


def _get_submission_for_student(conn, task_id: int, student_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT id, file_url, submitted_at, status
        FROM task_submissions
        WHERE task_id = ? AND student_id = ?
        """,
        (task_id, student_id),
    ).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "file_url": row["file_url"],
        "submitted_at": row["submitted_at"],
        "status": row["status"],
    }


@router.get("/students", response_model=List[StudentInfo])
async def list_students(current_user: Dict[str, Any] = Depends(get_current_user)) -> List[StudentInfo]:
    if current_user["role"] != "teacher":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Teachers only")
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT id, username FROM users WHERE role = 'student' AND status = 'active' ORDER BY username ASC",
        ).fetchall()
        return [StudentInfo(id=row["id"], username=row["username"]) for row in rows]
    finally:
        conn.close()


@router.get("/", response_model=List[TaskSummary])
async def list_tasks(current_user: Dict[str, Any] = Depends(get_current_user)) -> List[TaskSummary]:
    user_id = current_user["id"]
    role = current_user["role"]
    conn = get_db_connection()
    try:
        if role == "teacher":
            task_rows = conn.execute(
                """
                SELECT id, title, description, teacher_id, deadline, file_url, file_path, original_filename, created_at
                FROM tasks
                WHERE teacher_id = ?
                ORDER BY datetime(created_at) DESC
                """,
                (user_id,),
            ).fetchall()
        else:
            task_rows = conn.execute(
                """
                SELECT t.id, t.title, t.description, t.teacher_id, t.deadline, t.file_url, t.file_path, t.original_filename, t.created_at
                FROM tasks t
                JOIN task_assignees a ON a.task_id = t.id
                WHERE a.student_id = ?
                ORDER BY datetime(t.created_at) DESC
                """,
                (user_id,),
            ).fetchall()

        summaries: List[TaskSummary] = []
        now_utc = datetime.now(timezone.utc)

        for task in task_rows:
            task_id = task["id"]
            student_ids = _get_assignees(conn, task_id)
            deadline_iso = _iso_or_none(task["deadline"])
            file_download_url = f"/tasks/{task_id}/file" if task["file_path"] else None
            submission_status = None
            submitted_at = None
            is_overdue = None
            submitted_count = None
            total_assignees = None

            if role == "student":
                submission = _get_submission_for_student(conn, task_id, user_id)
                deadline_dt = _parse_datetime(task["deadline"])
                if submission:
                    submission_status = submission["status"]
                    submitted_at = _iso_or_none(submission["submitted_at"])
                    is_overdue = submission_status == "late"
                else:
                    overdue = deadline_dt is not None and now_utc > deadline_dt
                    submission_status = "overdue" if overdue else "pending"
                    is_overdue = overdue
            else:
                total_assignees = len(student_ids)
                count_row = conn.execute(
                    "SELECT COUNT(*) as count FROM task_submissions WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                submitted_count = count_row["count"] if count_row else 0

            summaries.append(
                TaskSummary(
                    id=task_id,
                    title=task["title"],
                    description=task["description"],
                    teacher_id=task["teacher_id"],
                    student_ids=student_ids,
                    deadline=deadline_iso,
                    file_url=file_download_url,
                    original_filename=task["original_filename"] if task["file_path"] else None,
                    created_at=_iso_or_none(task["created_at"]) or datetime.now(timezone.utc).isoformat(),
                    student_status=submission_status,
                    submitted_at=submitted_at,
                    is_overdue=is_overdue,
                    submitted_count=submitted_count,
                    total_assignees=total_assignees,
                )
            )

        return summaries
    finally:
        conn.close()


@router.post("/", response_model=TaskDetail, status_code=status.HTTP_201_CREATED)
async def create_task(
    title: str = Form(...),
    description: Optional[str] = Form(None),
    deadline: Optional[str] = Form(None),
    target_students: List[int] = Form(...),
    file: Optional[UploadFile] = File(None),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> TaskDetail:
    if current_user["role"] != "teacher":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Teachers only")
    if not target_students:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="学生を少なくとも1名選択してください")

    file_url = None
    file_path = None
    original_filename = None
    if file is not None:
        if not file.filename:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="アップロードファイル名が無効です")
        saved_path, saved_url, saved_name = _save_upload(file)
        file_path = saved_path
        file_url = saved_url
        original_filename = saved_name

    conn = get_db_connection()
    try:
        placeholders = ",".join(["?"] * len(target_students))
        rows = conn.execute(
            f"SELECT id FROM users WHERE id IN ({placeholders}) AND role = 'student' AND status = 'active'",
            tuple(target_students),
        ).fetchall()
        valid_student_ids = {row["id"] for row in rows}
        missing = set(target_students) - valid_student_ids
        if missing:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"存在しない学生ID: {sorted(missing)}")

        cursor = conn.execute(
            """
            INSERT INTO tasks (title, description, teacher_id, deadline, file_url, file_path, original_filename)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title.strip(),
                description.strip() if description else None,
                current_user["id"],
                deadline,
                file_url,
                file_path,
                original_filename,
            ),
        )
        task_id = cursor.lastrowid
        conn.executemany(
            "INSERT INTO task_assignees (task_id, student_id) VALUES (?, ?)",
            [(task_id, student_id) for student_id in target_students],
        )
        conn.commit()

        student_ids = _get_assignees(conn, task_id)
        students = _fetch_students(conn, student_ids)
        teacher_row = conn.execute(
            "SELECT username FROM users WHERE id = ?",
            (current_user["id"],),
        ).fetchone()
        teacher_name = teacher_row["username"] if teacher_row else None

        download_url = f"/tasks/{task_id}/file" if file_path else None

        return TaskDetail(
            id=task_id,
            title=title.strip(),
            description=description.strip() if description else None,
            teacher_id=current_user["id"],
            student_ids=student_ids,
            deadline=_iso_or_none(deadline),
            file_url=download_url,
            original_filename=original_filename,
            created_at=datetime.now(timezone.utc).isoformat(),
            teacher_name=teacher_name,
            students=students,
        )
    finally:
        conn.close()


@router.get("/{task_id}", response_model=TaskDetail)
async def get_task(task_id: int, current_user: Dict[str, Any] = Depends(get_current_user)) -> TaskDetail:
    conn = get_db_connection()
    try:
        task = conn.execute(
            """
            SELECT id, title, description, teacher_id, deadline, file_url, file_path, original_filename, created_at
            FROM tasks
            WHERE id = ?
            """,
            (task_id,),
        ).fetchone()
        if task is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

        teacher_id = task["teacher_id"]
        student_ids = _get_assignees(conn, task_id)

        role = current_user["role"]
        user_id = current_user["id"]
        if role == "teacher":
            if teacher_id != user_id:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
        elif role == "student":
            if user_id not in student_ids:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
        else:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

        teacher_row = conn.execute(
            "SELECT username FROM users WHERE id = ?",
            (teacher_id,),
        ).fetchone()
        teacher_name = teacher_row["username"] if teacher_row else None
        students = _fetch_students(conn, student_ids)

        download_url = f"/tasks/{task_id}/file" if task["file_path"] else None

        detail = TaskDetail(
            id=task["id"],
            title=task["title"],
            description=task["description"],
            teacher_id=teacher_id,
            student_ids=student_ids,
            deadline=_iso_or_none(task["deadline"]),
            file_url=download_url,
            original_filename=task["original_filename"] if task["file_path"] else None,
            created_at=_iso_or_none(task["created_at"]) or datetime.now(timezone.utc).isoformat(),
            teacher_name=teacher_name,
            students=students,
        )

        if role == "student":
            submission = _get_submission_for_student(conn, task_id, user_id)
            if submission:
                detail.student_status = submission["status"]
                detail.submitted_at = _iso_or_none(submission["submitted_at"]) if submission["submitted_at"] else None
        else:
            count_row = conn.execute(
                "SELECT COUNT(*) as count FROM task_submissions WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            detail.total_assignees = len(student_ids)
            detail.submitted_count = count_row["count"] if count_row else 0

        return detail
    finally:
        conn.close()


@router.post("/{task_id}/submit", response_model=SubmissionResponse, status_code=status.HTTP_201_CREATED)
async def submit_task(
    task_id: int,
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> SubmissionResponse:
    if current_user["role"] != "student":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Students only")
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="ファイルが選択されていません")

    conn = get_db_connection()
    try:
        task = conn.execute(
            "SELECT deadline FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if task is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

        assigned_rows = conn.execute(
            "SELECT 1 FROM task_assignees WHERE task_id = ? AND student_id = ?",
            (task_id, current_user["id"]),
        ).fetchall()
        if not assigned_rows:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="この課題には提出できません")

        file_path, file_url, original_filename = _save_upload(file)
        submitted_at = datetime.now(timezone.utc).isoformat()
        deadline_dt = _parse_datetime(task["deadline"])
        status_value = "submitted"
        if deadline_dt is not None and datetime.now(timezone.utc) > deadline_dt:
            status_value = "late"

        conn.execute(
            """
            INSERT INTO task_submissions (task_id, student_id, file_url, submitted_at, status)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(task_id, student_id)
            DO UPDATE SET file_url = excluded.file_url, submitted_at = excluded.submitted_at, status = excluded.status
            """,
            (task_id, current_user["id"], file_url, submitted_at, status_value),
        )
        conn.commit()

        return SubmissionResponse(
            id=None,
            student_id=current_user["id"],
            student_name=current_user.get("username", ""),
            file_url=file_url,
            status=status_value,
            submitted_at=submitted_at,
        )
    finally:
        conn.close()


@router.get("/{task_id}/submissions", response_model=List[SubmissionResponse])
async def list_submissions(task_id: int, current_user: Dict[str, Any] = Depends(get_current_user)) -> List[SubmissionResponse]:
    if current_user["role"] != "teacher":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Teachers only")

    conn = get_db_connection()
    try:
        task = conn.execute(
            "SELECT teacher_id, deadline FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if task is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
        if task["teacher_id"] != current_user["id"]:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

        deadline_dt = _parse_datetime(task["deadline"])
        student_ids = _get_assignees(conn, task_id)
        students = _fetch_students(conn, student_ids)
        student_lookup = {student.id: student.username for student in students}

        submissions: List[SubmissionResponse] = []
        for student_id in student_ids:
            submission = _get_submission_for_student(conn, task_id, student_id)
            if submission:
                submissions.append(
                    SubmissionResponse(
                        id=submission["id"],
                        student_id=student_id,
                        student_name=student_lookup.get(student_id, ""),
                        file_url=submission["file_url"],
                        status=submission["status"],
                        submitted_at=_iso_or_none(submission["submitted_at"]),
                    )
                )
            else:
                overdue = deadline_dt is not None and datetime.now(timezone.utc) > deadline_dt
                submissions.append(
                    SubmissionResponse(
                        id=None,
                        student_id=student_id,
                        student_name=student_lookup.get(student_id, ""),
                        file_url=None,
                        status="overdue" if overdue else "pending",
                        submitted_at=None,
                    )
                )

        return submissions
    finally:
        conn.close()


@router.get("/{task_id}/file")
async def download_task_file(task_id: int, current_user: Dict[str, Any] = Depends(get_current_user)) -> FileResponse:
    conn = get_db_connection()
    try:
        task_row = conn.execute(
            "SELECT teacher_id, file_path, original_filename FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if task_row is None or not task_row["file_path"]:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not available")

        teacher_id = task_row["teacher_id"]
        if current_user["role"] == "teacher":
            if teacher_id != current_user["id"]:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
        elif current_user["role"] == "student":
            assigned = conn.execute(
                "SELECT 1 FROM task_assignees WHERE task_id = ? AND student_id = ?",
                (task_id, current_user["id"]),
            ).fetchone()
            if not assigned:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
        else:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

        file_path = Path(task_row["file_path"])
        if not file_path.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

        filename = task_row["original_filename"] or file_path.name
        return FileResponse(path=file_path, filename=filename)
    finally:
        conn.close()
