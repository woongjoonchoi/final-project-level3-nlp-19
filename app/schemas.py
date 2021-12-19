from pydantic import BaseModel
from uuid import UUID


class Question(BaseModel):

    id: str
    text: str