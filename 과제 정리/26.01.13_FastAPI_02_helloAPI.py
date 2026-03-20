from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

class HelloRequest(BaseModel):
    name: str = Field(...)

class HelloResponse(BaseModel):
    message: str

@app.post("/hello", response_model=HelloResponse)
def hello(body: HelloRequest):
    if body.name.strip() == "":
        return {"message": "이름을 입력하세요"}
    return {
        "message": f"안녕하세요, {body.name}님!"
    }
