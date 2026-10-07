# Document Chat Bot

A chatbot built with LangChain and Gemini. You upload a document and ask questions about it. It also has two tools: a calculator and a weather lookup (Open-Meteo).

It reads PDF, Word, text, CSV and image files. Scanned PDFs and PDFs made with "Microsoft Print to PDF" have no text layer, so the bot renders each page as an image and reads it with Gemini OCR. If one Gemini model is overloaded, it tries the next one.

## 🚀 Live Demo
Try it here: [Readify Chatbot](https://readifychatbot.streamlit.app/)

## Setup

1. Clone the repo and create a virtual environment:

   ```
   git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
   cd YOUR-REPO
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Get a Gemini API key from Google AI Studio, then copy `.env.example` to `.env` and paste the key in:

   ```
   GOOGLE_API_KEY=your-key-here
   ```

## Run

Web interface:

```
streamlit run app.py
```

Terminal version (type `load <file path>`, `unload`, or `quit`):

```
python agent.py
```

## Files

- `agent.py` holds the model, tools, OCR and document loading.
- `app.py` is the Streamlit interface. It imports `agent.py`.

## Notes

- Gemini model names change often. If you get a 404, list the models your key can use and update the names in `agent.py` (`llm` and `OCR_MODELS`).
- Chat memory keeps the last 5 messages (`MEMORY_SIZE` in `agent.py`).
- The calculator uses `eval`, which is fine for local use but not safe to expose to untrusted users.
"# readify_agent_bot" 
"# readify_agent_bot" 
