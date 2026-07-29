from fastapi import Depends, FastAPI

app = FastAPI()


def get_db():
    return "database_connection"


@app.get("/items/")
def read_items(
    db = Depends(get_db),  # Noncompliant {{Use "Annotated" type hints for FastAPI dependency injection}}
    session = Depends(get_db),  # Noncompliant
):
    return {"db": db, "session": session}
