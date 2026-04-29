from sqlalchemy.orm import Session

from app.models.business import Business
from app.schemas.business import BusinessCreate


class BusinessRepository:
    @staticmethod
    def create(db: Session, payload: BusinessCreate) -> Business:
        business = Business(**payload.model_dump())
        db.add(business)
        db.commit()
        db.refresh(business)
        return business

    @staticmethod
    def get(db: Session, business_id: int) -> Business | None:
        return db.query(Business).filter(Business.id == business_id).first()
