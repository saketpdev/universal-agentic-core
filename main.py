import logging
from fastapi import FastAPI
from api.routes import router
import uvicorn

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Universal Agentic Core", version="1.0.0")

app.include_router(router)

@app.get("/health")
def health_check():
    return {"status": "operational", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)