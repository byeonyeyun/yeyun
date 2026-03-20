# 검색 API 만들기
# GET /products/search?q=apple&limit=5
# 응답: {"q": "apple", "limit": 5}

from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/products/search")
def search(q: str, limit: int):
    return {"q": q, "limit": limit}
