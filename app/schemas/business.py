from pydantic import BaseModel, Field


class BusinessCreate(BaseModel):
    name: str = Field(..., min_length=1)
    business_type: str
    location: str | None = None
    notes: str | None = None


class BusinessRead(BusinessCreate):
    id: int

    class Config:
        from_attributes = True
