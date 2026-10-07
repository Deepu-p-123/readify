import os
import time
import tempfile
import hashlib
import base64
import warnings
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

warnings.filterwarnings("ignore")  # hide library warnings

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

MEMORY_SIZE = 5             # number of recent chat messages the bot remembers
MAX_DOC_CHARS = 200_000     # safety limit on document size

# OCR tries these models in order; if one is overloaded (503) it moves on
OCR_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.6-flash"]

# Chat model (fast)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    timeout=60,
    max_retries=1,
)

# OCR models (we handle retries ourselves, so max_retries=0)
ocr_llms = [
    ChatGoogleGenerativeAI(
        model=m,
        google_api_key=api_key,
        timeout=90,
        max_retries=0,
    )
    for m in OCR_MODELS
]


# ---------- Helpers ----------
def status_hook(message: str):
    """Progress messages go here. The Streamlit app replaces this function."""
    print(f"Bot: {message}")


def get_text(output):
    """Gemini may return a list of content blocks instead of a plain string."""
    if isinstance(output, str):
        return output
    return "".join(
        block.get("text", "") for block in output if block.get("type") == "text"
    )


# ---------- Tools ----------
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression like '2 + 3 * 4' and return the result."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception:
        return "I couldn't calculate that expression."


@tool
def get_weather(city: str) -> str:
    """Get the current temperature and humidity for a given city."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=5,
        ).json()
        if "results" not in geo:
            return f"Could not find the city {city}."
        loc = geo["results"][0]

        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "current": "temperature_2m,relative_humidity_2m",
            },
            timeout=5,
        ).json()["current"]
        return (
            f"Weather in {city}: {weather['temperature_2m']}°C, "
            f"humidity {weather['relative_humidity_2m']}%."
        )
    except Exception as e:
        return f"Weather service error: {e}"


tools = [calculator, get_weather]


# ---------- OCR with Gemini ----------
OCR_INSTRUCTION = (
    "Extract all the text from this page exactly as written, in reading "
    "order. Keep tables as plain text rows. Return only the extracted text, "
    "with no commentary."
)

_ocr_cache = {}  # remembers finished pages so a retry skips them


def ocr_image_bytes(img_bytes: bytes, mime_type: str = "image/png") -> str:
    """Send one image to Gemini, trying other models if one is overloaded."""
    data = base64.b64encode(img_bytes).decode("utf-8")
    message = HumanMessage(content=[
        {"type": "text", "text": OCR_INSTRUCTION},
        {"type": "image_url",
         "image_url": {"url": f"data:{mime_type};base64,{data}"}},
    ])
    last_error = None
    for round_no in range(4):
        for model in ocr_llms:
            try:
                return get_text(model.invoke([message]).content)
            except Exception as e:
                last_error = e
        time.sleep(3 * (2 ** round_no))  # wait 3, 6, 12, 24 seconds
    raise RuntimeError(f"OCR failed: {last_error}")


def ocr_pdf(path: str) -> str:
    """Render each PDF page to an image and OCR it."""
    import pymupdf
    parts, failed = [], []
    with pymupdf.open(path) as pdf:
        total = len(pdf)
        for i, page in enumerate(pdf, 1):
            png = page.get_pixmap(dpi=150).tobytes("png")
            key = hashlib.md5(png).hexdigest()  # same page content = cache hit
            if key not in _ocr_cache:
                status_hook(f"OCR page {i}/{total}...")
                try:
                    _ocr_cache[key] = ocr_image_bytes(png)
                except Exception:
                    failed.append(i)
                    parts.append(f"[Page {i} could not be read]")
                    continue
            parts.append(_ocr_cache[key])
    if failed:
        status_hook(f"Pages {failed} failed. Load the file again to retry "
                    "only those pages.")
    return "\n\n".join(parts)


def ocr_with_gemini(path: str, mime_type: str) -> str:
    """OCR a PDF or an image file."""
    if mime_type == "application/pdf":
        return ocr_pdf(path)
    with open(path, "rb") as f:
        return ocr_image_bytes(f.read(), mime_type)


# ---------- Document loading ----------
IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def read_pdf(path: str) -> str:
    """Try normal text extraction first, then OCR if the PDF is scanned."""
    text = ""

    # 1) pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        pass

    # 2) pymupdf (handles more PDFs)
    if len(text.strip()) < 50:
        try:
            import pymupdf
            with pymupdf.open(path) as pdf:
                text = "\n".join(page.get_text() for page in pdf)
        except Exception:
            pass

    # 3) OCR with Gemini (scanned or "Print to PDF" files)
    if len(text.strip()) < 50:
        status_hook("No text layer found, using OCR (this may take a moment)...")
        text = ocr_with_gemini(path, "application/pdf")

    return text


def read_document(path: str) -> str:
    """Read text from txt, md, csv, pdf, docx, or image files."""
    ext = os.path.splitext(path)[1].lower()

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    if ext in (".txt", ".md", ".csv"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if ext == ".pdf":
        return read_pdf(path)

    if ext == ".docx":
        from docx import Document
        doc = Document(path)
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:  # include text inside tables
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(parts)

    if ext in IMAGE_TYPES:
        status_hook("Reading the image with OCR...")
        return ocr_with_gemini(path, IMAGE_TYPES[ext])

    raise ValueError(f"Unsupported file type: {ext}")


def read_document_bytes(filename: str, data: bytes) -> str:
    """Read an uploaded file (name + raw bytes), e.g. from the web interface."""
    suffix = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return read_document(tmp_path)
    finally:
        os.remove(tmp_path)


# ---------- Agent ----------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Use the tools when needed.\n\n"
        "The user may have loaded a document. When a question is about the "
        "document, answer using its content. If the answer is not in the "
        "document, say so instead of guessing.\n\n"
        "DOCUMENT:\n{document}",
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)


# ---------- Chat loop ----------
def main():
    chat_history = []
    document_text = "(no document loaded)"

    print("Commands: load <file path> | unload | quit")
    print("Supported: .txt .md .csv .pdf .docx .png .jpg .jpeg .webp "
          "(scans use OCR)")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit"]:
            break

        # Load a document
        if user_input.lower().startswith("load "):
            path = user_input[5:].strip().strip('"').strip("'")
            try:
                text = read_document(path)
                if not text.strip():
                    print("Bot: I couldn't find any text in that file.")
                    continue
                if len(text) > MAX_DOC_CHARS:
                    text = text[:MAX_DOC_CHARS]
                    print(f"Bot: The document is long, so I only loaded the "
                          f"first {MAX_DOC_CHARS:,} characters.")
                document_text = text
                chat_history = []  # start fresh for the new document
                print(f"Bot: Loaded '{os.path.basename(path)}' "
                      f"({len(text):,} characters). Ask me anything about it.")
            except FileNotFoundError:
                print("Bot: I couldn't find that file. Check the path.")
            except Exception as e:
                print(f"Bot: I couldn't read that file ({e}).")
            continue

        # Remove the document
        if user_input.lower() == "unload":
            document_text = "(no document loaded)"
            chat_history = []
            print("Bot: Document removed.")
            continue

        # Normal question
        for attempt in range(3):
            try:
                result = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": chat_history,
                    "document": document_text,
                })
                answer = get_text(result["output"])
                print("Bot:", answer)

                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=answer))
                chat_history = chat_history[-MEMORY_SIZE:]

                # History must start with a human message
                if chat_history and isinstance(chat_history[0], AIMessage):
                    chat_history = chat_history[1:]
                break
            except Exception as e:
                if "503" in str(e) and attempt < 2:
                    time.sleep(2 ** attempt)  # silent retry
                else:
                    print("Bot: Sorry, something went wrong. Please try again.")
                    break


if __name__ == "__main__":
    main()