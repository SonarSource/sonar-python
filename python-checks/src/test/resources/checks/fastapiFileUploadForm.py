from typing import List, Optional
from fastapi import FastAPI, APIRouter, File, Body, Depends, Form, UploadFile
from pydantic import BaseModel
from starlette.datastructures import UploadFile as StarletteUploadFile

app = FastAPI()
router = APIRouter()


class Base(BaseModel):
    name: str
    sensitive_data: str


class PolicyData(BaseModel):
    policy_id: str
    details: dict


class DataConfig(BaseModel):
    config: str


@router.post("/upload")
async def upload_with_body(
    country_id: str = Body(...),  # Noncompliant {{Use "Form()" instead of "Body()" when handling file uploads; "Body()" expects JSON, which is incompatible with multipart/form-data.}}
    policy_details: List[dict] = Body(...),  # Noncompliant
    files: List[UploadFile] = File(...)
):
    return {"status": "ok"}


@app.post("/upload2")
def upload_single_body(
    data: dict = Body(...),  # Noncompliant
    file: UploadFile = File(...)
):
    pass


@app.post("/submit")
def submit_with_depends(
    base: Base = Depends(),  # Noncompliant {{Use "Form()" with Pydantic validators instead of "Depends()" for file upload endpoints; query parameters may expose sensitive data in URLs.}}
    files: List[UploadFile] = File(...)
):
    pass


@router.post("/submit2")
async def submit_explicit_depends(
    data: PolicyData = Depends(PolicyData),  # Noncompliant
    file: UploadFile = File(...)
):
    pass


@router.post("/upload-compliant")
async def compliant_upload(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    pass


@app.post("/json-only")
def json_endpoint(
    data: PolicyData = Body(...),
):
    pass


@router.post("/no-file")
def no_file_depends(
    base: Base = Depends(),
):
    pass


def get_current_user():
    return {"user": "test"}


@app.post("/depends-function")
def depends_with_function(
    user = Depends(get_current_user),
    file: UploadFile = File(...)
):
    pass


def parse_config(data: str = Form(...)) -> DataConfig:
    return DataConfig.model_validate_json(data)


@app.post("/data")
async def upload_data(
    config: DataConfig = Depends(parse_config),
    csv_file: UploadFile = File(...)
):
    pass


@router.post("/multi")
def multi_decorator(
    data: str = Body(...),  # Noncompliant
    file: UploadFile = File(...)
):
    pass


@app.post("/optional")
def optional_file(
    data: str = Body(...),  # Noncompliant
    file: Optional[UploadFile] = File(None)
):
    pass


@app.post("/multiple-files")
def multiple_files(
    data: str = Body(...),  # Noncompliant
    files: List[UploadFile] = File(...)
):
    pass


@app.post("/sync")
def sync_fn(data: str = Body(...), file: UploadFile = File(...)): pass  # Noncompliant


@app.post("/async")
async def async_fn(data: str = Body(...), file: UploadFile = File(...)): pass  # Noncompliant


def regular_function(
    data: str = Body(...),
    file: UploadFile = File(...)
):
    pass


@app.put("/update")
def put_endpoint(data: str = Body(...), file: UploadFile = File(...)): pass  # Noncompliant


@app.patch("/patch")
def patch_endpoint(data: str = Body(...), file: UploadFile = File(...)): pass  # Noncompliant


@app.delete("/delete-with-file")
def delete_endpoint(data: str = Body(...), file: UploadFile = File(...)): pass  # Noncompliant


@app.post("/file-only")
def file_only_endpoint(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    pass


@app.post("/mixed")
def mixed_params(
    valid_field: str = Form(...),
    invalid_field: str = Body(...),  # Noncompliant
    file: UploadFile = File(...)
):
    pass


@app.post("/both-issues")
def both_issues(
    body_data: str = Body(...),  # Noncompliant
    model_data: Base = Depends(),  # Noncompliant
    file: UploadFile = File(...)
):
    pass


@app.post("/type-annotation-only")
def type_annotation_file_without_default_value(
    file: UploadFile,  # No File() default, just type annotation
    data: str = Body(...)  # Noncompliant
):
    pass


@app.post("/path-param/{item_id}")
def path_param_with_file(
    item_id: str,  # No default value - path parameter
    data: str = Body(...),  # Noncompliant
    file: UploadFile = File(...)
):
    pass


@app.post("/depends-no-annotation")
def depends_no_annotation(
    dep = Depends(get_current_user),  # No type annotation - compliant
    file: UploadFile = File(...)
):
    pass


@app.post("/starlette-file")
def starlette_upload(
    data: str = Body(...),  # Noncompliant
    file: StarletteUploadFile = File(...)
):
    pass


# Edge: Depends() with unpacking argument 
def dependency_func():
    return {"key": "value"}

dependency_args = [dependency_func]

@app.post("/depends-unpacking-arg")
def depends_unpacking_arg(
    dep = Depends(*dependency_args),  # unpacking argument - compliant
    file: UploadFile = File(...)
):
    pass


@app.post("/empty-depends-with-annotation")
def empty_depends_with_annotation(
    base: Base = Depends(),  # Noncompliant
    file: UploadFile = File(...)
):
    pass

def null_parameterlist():
    from fastapi import APIRouter

    router = APIRouter()


    @router.post("/")
    async def update_admin():
        ...
