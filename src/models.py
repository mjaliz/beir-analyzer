from pydantic import BaseModel, Field, TypeAdapter


class Corpus(BaseModel):
    id: str = Field(alias="_id")
    text: str
    title: str
    q_id: str | None = Field(default=None)
    q_text: str | None = Field(default=None)
    qrel_score: int | None = Field(default=None)


class Query(BaseModel):
    id: str = Field(alias="_id")
    text: str


corups_list_adaptor = TypeAdapter(list[Corpus])
