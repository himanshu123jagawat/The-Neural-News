from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

# Environment variables se API keys lo
API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY", "AIzaSyDLiJpvyILhC-YoDZvXqhxsGWe1q2aHCGg")
CSE_ID = os.getenv("GOOGLE_CSE_ID", "6477905260a454e0e")

app = FastAPI(
    title="Evidence Retrieval API",
    description="This API's only job is to find evidence from trusted sources.",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Neural News Search API is running"}

@app.get("/verify")
async def find_evidence(q: str):
    """
    This endpoint searches for a query and returns the raw evidence it finds.
    """
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required.")

    url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={CSE_ID}&q={q}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        results = response.json()

        if 'items' not in results or int(results.get('searchInformation', {}).get('totalResults', 0)) == 0:
            return {
                "status": "NO_EVIDENCE_FOUND",
                "evidence": []
            }

        evidence = []
        for item in results.get('items', []):
            evidence.append({
                "title": item.get('title'),
                "snippet": item.get('snippet'),
                "source": item.get('displayLink')
            })
        
        return {
            "status": "EVIDENCE_FOUND",
            "evidence": evidence[:4]
        }

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"API connection error: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)