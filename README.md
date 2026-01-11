# Multi-Modal Local RAG

End-to-end RAG (Retrieval-Augmented Generation) system with **LangGraph** backend for PDF ingestion, chunking, FAISS vector indexing, and retrieval-augmented QA using Ollama models. Features **Agent Chat UI** for a modern chat interface with streaming responses.

## Features

- **PDF Processing**: Advanced PDF ingestion with text chunking, table extraction, and image handling
- **Vector Search**: FAISS-based vector indexing and similarity search
- **Multi-Modal Support**: Handles text, tables, and images from PDF documents
- **LangGraph Agent**: Tool-calling agent with document ingestion and search capabilities
- **Streaming Responses**: Real-time streaming responses via LangGraph
- **Modern Chat UI**: Agent Chat UI (Next.js) for interactive conversations
- **Persistent Storage**: JSON-based document store with vector persistence

## Prerequisites

- Python ≥3.11
- Node.js ≥18
- Ollama running locally with required models

## Quick Start

### 1. Install Python Dependencies

```bash
# Install uv if not already installed
pip install uv

# Install dependencies
uv sync
```

### 2. Install Ollama Models

Ensure Ollama is running and pull required models:

```bash
ollama pull gemma3
ollama pull embeddinggemma:300m
```

### 3. Install LangGraph CLI

```bash
pip install -U "langgraph-cli[inmem]"
```

### 4. Start the LangGraph Server

```bash
langgraph dev
```

The LangGraph server will be available at `http://localhost:2024`.

### 5. Start the Agent Chat UI

In a new terminal:

```bash
cd agent-chat-ui
npm run dev
```

The chat UI will be available at `http://localhost:3000`.

### 6. Connect and Chat

1. Open `http://localhost:3000` in your browser
2. Enter:
   - **Deployment URL**: `http://localhost:2024`
   - **Assistant/Graph ID**: `rag_agent`
3. Click **Continue** to start chatting

## Usage

### Ingesting Documents

In the chat, ask the agent to ingest a document:

```
Please ingest the document at /path/to/your/document.pdf
```

### Asking Questions

After ingesting documents, ask questions:

```
What are the main topics covered in the document?
```

The agent will automatically search the knowledge base and provide answers based on the retrieved context.

## Configuration

### Environment Variables (.env)

```env
APP_ENV=local
DATA_DIR=./storage
EMBEDDING_MODEL=embeddinggemma:300m
CHAT_MODEL=gemma3
OLLAMA_BASE_URL=http://localhost:11434
SEARCH_K=4
LOG_LEVEL=INFO
LOG_TO_FILE=false
```

### Agent Chat UI Environment

Create `agent-chat-ui/apps/web/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:2024
NEXT_PUBLIC_ASSISTANT_ID=rag_agent
```

## Project Structure

```
├── backend/
│   ├── langgraph/             # LangGraph components
│   │   ├── graph.py           # Main RAG agent graph
│   │   ├── state.py           # Graph state definitions
│   │   └── tools.py           # Agent tools (ingest, search)
│   ├── core/
│   │   ├── config.py          # Environment configuration
│   │   └── dependency.py      # Dependency injection
│   ├── models/
│   │   └── schemas.py         # Pydantic models
│   ├── servies/               # Business logic services
│   │   ├── chat_service.py    # RAG orchestration
│   │   ├── file_service.py    # PDF processing
│   │   └── model_service.py   # Ollama model wrappers
│   ├── system_prompts/        # System prompts
│   └── utils/                 # Utilities
├── agent-chat-ui/             # Agent Chat UI (Next.js)
├── langgraph.json             # LangGraph server config
├── pyproject.toml             # Python dependencies
└── .env                       # Environment configuration
```

## Tech Stack

- **Backend**: LangGraph, LangChain, Ollama
- **Vector Store**: FAISS
- **Document Processing**: Unstructured, PDF2Image
- **Frontend**: Agent Chat UI (Next.js, React)
- **ML/AI**: Transformers, PyTorch

## Available Tools

The RAG agent has access to these tools:

| Tool                    | Description                                              |
| ----------------------- | -------------------------------------------------------- |
| `ingest_document`       | Upload and process PDF documents into the knowledge base |
| `search_knowledge_base` | Search for relevant information in uploaded documents    |

## Development

### Run LangGraph in Development Mode

```bash
langgraph dev --verbose
```

### View LangGraph Studio

When running `langgraph dev`, LangGraph Studio is available at `http://localhost:2024/studio` for debugging and visualization.

## Troubleshooting

### Ollama Connection Issues

Ensure Ollama is running:

```bash
ollama serve
```

### Vector Store Issues

The vector store is persisted in `./storage/vector_store/`. To reset:

```bash
rm -rf ./storage/vector_store ./storage/docstore.json
```

### Agent Chat UI Issues

If the UI can't connect, verify:

1. LangGraph server is running on port 2024
2. Environment variables are correctly set
3. CORS is not blocking requests (local development should work automatically)
