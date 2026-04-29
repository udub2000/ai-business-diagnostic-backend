from app.db.base import Base
from app.db.session import engine
from app.models.business import Business
from app.models.assessment import Assessment


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
