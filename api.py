from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
# Import your new streaming function
from fast_rag import stream_rag_pipeline 

app = FastAPI(title="AxIn Help RAG API")

class Query(BaseModel):
    question: str

@app.post("/stream")
def stream_question(query: Query):
    print(f"\n[RECEIVED STREAM REQUEST] Question: {query.question}")
    
    try:
        # Pass the generator directly into the StreamingResponse
        # media_type="text/event-stream" tells the client to keep the connection open
        return StreamingResponse(
            stream_rag_pipeline(query.question), 
            media_type="text/event-stream"
        )
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail="Internal RAG Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)