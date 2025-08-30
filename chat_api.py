from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
import json
import google.generativeai as genai
import os

# Environment variables se API key lo
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDzobqwJRoQjoACusc1p2u28la7c_m_8Wc")

app = FastAPI(title="Neural News Chat API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Gemini configuration error: {e}")
    gemini_model = None

# Chat History
chat_history = []

class QueryRequest(BaseModel):
    text: str

def call_gemini_flash(prompt, history):
    if not gemini_model:
        return "Gemini model not loaded. Check API Key."
    try:
        generation_config = genai.types.GenerationConfig(temperature=0.2)
        chat_session = gemini_model.start_chat(history=history)
        response = chat_session.send_message(prompt, generation_config=generation_config)
        return response.text
    except Exception as e:
        return f"Gemini API error: {e}"

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.post("/analyze-with-agent")
async def analyze_with_agent(request: QueryRequest):
    global chat_history
    
    user_query = request.text
    
    # Translate query for search
    search_query_prompt = f"Translate the following user query to English so it can be used for a web search. If it's already in English, just repeat it. Query: '{user_query}'"
    translated_query = call_gemini_flash(search_query_prompt, [])
    
    try:
        # Search service URL - pointing to the same domain's search endpoint
        search_api_url = os.getenv("SEARCH_API_URL", "https://theneuralnews.info")
        fact_checker_url = f"{search_api_url}/verify?q={translated_query}"
        response = requests.get(fact_checker_url)
        response.raise_for_status()
        api_result = response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Cannot connect to fact-checking service: {e}")
    
    # Build evidence string
    evidence_str = ""
    if api_result.get("evidence"):
        for item in api_result["evidence"]:
            evidence_str += f"- Source: {item.get('source', 'N/A')}\n  Title: {item.get('title', 'N/A')}\n  Snippet: {item.get('snippet', 'N/A')}\n\n"
    
    # Master prompt for Gemini
    master_prompt = f"""
    You are 'Veritas', a smart, multilingual, and expert AI fact-checker.

    **Conversation History:**
    {json.dumps(chat_history[-4:], indent=2)}

    **User's Latest Message (in their original language):**
    "{user_query}"

    **Evidence Retrieved from Trusted Sources (in English):**
    Status: {api_result['status']}
    {evidence_str if evidence_str else "No evidence was found."}

    **Your Task (Follow these steps):**
    1.  **Analyze Intent:** Understand the user's latest message in its original language. Is it a fact-checking question or a simple conversational remark (like a greeting)?
    2.  **Analyze Evidence:** Look at the retrieved evidence. Is it relevant to the user's question?
    3.  **Decide Action:** Choose ONE of the following actions:

        * **Action: Chit-Chat:** If the user's intent is conversational, IGNORE the evidence and provide a short, friendly, conversational reply IN THE USER'S ORIGINAL LANGUAGE.

        * **Action: Fact-Check:** If the user's intent is to get information verified, perform a fact-check. Follow this structure:
            a. **Verdict:** Start with a clear verdict (True, False, or Uncertain).
            b. **Explanation:** In 1-2 simple sentences, explain your reasoning based on the evidence.
            c. **Sources:** List the source platforms you analyzed.
            d. **Translate:** Ensure your entire final answer is in the USER'S ORIGINAL LANGUAGE.
    """
    
    final_answer = call_gemini_flash(master_prompt, chat_history)
    
    # Update chat history
    chat_history.append({'role': 'user', 'parts': [user_query]})
    chat_history.append({'role': 'model', 'parts': [final_answer]})
    if len(chat_history) > 6:
        chat_history = chat_history[-6:]
    
    return {"agent_response": final_answer}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)