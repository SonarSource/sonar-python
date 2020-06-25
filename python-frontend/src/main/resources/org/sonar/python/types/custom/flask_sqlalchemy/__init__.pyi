from sqlalchemy.engine.base import Engine
from sqlalchemy.orm.session import Session

class SQLAlchemy(CustomStubBase):
    engine: Engine
    session: Session
