from typing import Annotated
from fastapi import Depends, FastAPI, Query

app = FastAPI()


def get_db():
    return "database_connection"


@app.get("/items/")
def read_items(
    db = Depends(get_db),
    limit: int = Query(10),
    page: int = Query(1),
    search: Annotated[str | None, Query(max_length=50)] = None,
):
    return {"db": db, "search": search, "limit": limit, "page": page}
