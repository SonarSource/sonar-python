from typing import Annotated
from fastapi import Depends, FastAPI, Query

app = FastAPI()


def get_db():
    return "database_connection"


@app.get("/items/")
def read_items(
    db: Annotated[str, Depends(get_db)],
    search: Annotated[str | None, Query(max_length=50)] = None,
    limit: int = Query(10),  # Noncompliant {{Use "Annotated" type hints for FastAPI dependency injection}}
):
    return {"db": db, "search": search, "limit": limit}
