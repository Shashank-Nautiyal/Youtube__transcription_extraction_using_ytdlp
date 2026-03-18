# YouTube Playlist RAG Q&A

A Retrieval-Augmented Generation (RAG) pipeline that lets you ask questions about the content of a YouTube playlist. It downloads audio from playlist videos, transcribes them using OpenAI Whisper, stores the transcripts in a vector database, and answers natural-language questions using a local LLM via Ollama.

---

## How It Works

```
YouTube Playlist → yt-dlp (audio) → Whisper (transcription) → JSON
       → LangChain (chunking) → ChromaDB (vector store) → Ollama LLM → Answer
```

1. **Fetch video IDs** from a YouTube playlist using the YouTube Data API v3.
2. **Download audio** for each video using `yt-dlp`.
3. **Transcribe audio** using OpenAI Whisper (`medium` model, GPU-accelerated if available).
4. **Save transcripts** to a `transcripts.json` file.
5. **Load & chunk** the transcripts using LangChain's `RecursiveCharacterTextSplitter` (chunk size: 800, overlap: 150).
6. **Embed chunks** using `qwen3-embedding` via Ollama and store them in a local ChromaDB vector store.
7. **Answer questions** by retrieving the top-4 relevant chunks and passing them to `qwen3.5:9b` via a strict context-only prompt.

---

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com/) running locally with the following models pulled:
  - `qwen3-embedding:latest`
  - `qwen3.5:9b`
- A Google Cloud project with the **YouTube Data API v3** enabled and a valid API key
- A CUDA-capable GPU (optional but recommended for Whisper)

---

## Installation

```bash
pip install langchain langchain-community langchain-ollama langchain-text-splitters \
            chromadb google-api-python-client yt-dlp openai-whisper torch
```

Pull the required Ollama models:

```bash
ollama pull qwen3-embedding:latest
ollama pull qwen3.5:9b
```

---

## Configuration

Open `youtube_transcript.ipynb` and set your credentials in the relevant cells:

```python
API_KEY = "YOUR_YOUTUBE_DATA_API_KEY"
PLAYLIST_ID = "YOUR_PLAYLIST_ID"
```

> ⚠️ **Do not commit your API key to version control.** Consider using environment variables or a `.env` file instead.

You can also control how many videos to process:

```python
selected_videos = video_ids[0:33]  # adjust the slice as needed
```

---

## Usage

Run the notebook cells in order:

| Step | What it does |
|------|-------------|
| Cell 1 | Imports |
| Cell 2 | Fetches all video IDs from the playlist |
| Cell 3 | Selects a subset of videos to process |
| Cell 4 | Downloads audio & transcribes with Whisper → saves `transcripts.json` |
| Cell 5–6 | Loads transcripts with LangChain `JSONLoader` |
| Cell 7 | Splits transcripts into chunks |
| Cell 8 | Embeds chunks and stores them in ChromaDB (`./new_chroma_db`) |
| Cell 9 | Defines the `retriever()` function |
| Cell 10 | Loads the `qwen3.5:9b` chat model |
| Cell 11–14 | Builds the RAG chain (retriever → prompt → model → parser) |
| Cell 15 | Runs a sample query |

To ask a question:

```python
question = "can you summarize the video"
chain.invoke(question)
```

---

## Project Structure

```
.
├── youtube_transcript.ipynb   # Main notebook
├── transcripts.json           # Generated: raw transcripts (auto-created)
└── new_chroma_db/             # Generated: persistent ChromaDB vector store (auto-created)
```

---

## Notes

- The Whisper `medium` model offers a good balance between speed and accuracy. You can swap it for `small`, `large`, etc.
- The RAG prompt is intentionally strict — the LLM is instructed to answer only from the retrieved context and will reply "I don't know." if the context is insufficient.
- Transcription is the most time-intensive step; a GPU significantly speeds it up.
- The vector store persists to disk, so embeddings only need to be generated once.
