from pydantic import BaseModel, Field


class KnowledgeEntry(BaseModel):
    id: str
    symptoms: list[str]
    condition: str
    advice: str
    urgent: bool = False
    disclaimer: str = "This information is educational only and is not a medical diagnosis."


class SymptomLog(BaseModel):
    id: int
    user_id: str
    timestamp: str
    symptoms: str
    predicted_condition: str
    advice_given: str


class Appointment(BaseModel):
    id: int
    user_id: str
    requested_at: str
    reason: str
    preferred_date: str
    status: str = Field(default="pending")
