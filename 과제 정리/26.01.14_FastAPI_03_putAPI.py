from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

items = [
    {"id": 1, "name": "apple", "price": 100},
    {"id": 2, "name": "banana", "price": 80},
    {"id": 3, "name": "cherry", "price": 50},
]
class ItemResponse(BaseModel):
    id: int
    name: str
    price: int

class ItemUpdateRequest(BaseModel):
    name: str
    price: int

@app.put("/items/{item_id}")
def put_item_api(item_id: int, body: ItemUpdateRequest) -> ItemResponse:
    for item in items:
        if item["id"] == item_id:
            item["name"] = body.name
            item["price"] = body.price
            return item

    raise HTTPException(
        status_code=404,
        detail=f"Item Not Found(id:{item_id})"
    )
