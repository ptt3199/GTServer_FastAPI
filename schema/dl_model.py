from pydantic import BaseModel


class ModelInputSchema(BaseModel):
    request_id: str
    model_name: str
    img_dir: str


class ModelOutputSchema(BaseModel):
    output: int