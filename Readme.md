
# RAG Chatbot with Qdrant

A Retrieval Augmented Generation (RAG) chatbot with FastAPI backend.

## Project Structure

The project has been refactored to improve organization and maintainability:

```
app/
├── config/                 # Configuration settings
│   ├── __init__.py
│   ├── settings.py         # Centralized settings
│   └── gpu_config.py       # GPU configuration
├── models/                 # Pydantic models
│   ├── __init__.py
│   ├── user_models.py      # User-related models
│   └── chatbot_models.py   # Chatbot-related models
├── database/               # Database operations
│   ├── __init__.py
│   ├── mysql_db.py         # MySQL database operations
│   └── vector_db.py        # Vector database operations
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── chat_history.py     # Chat history operations
│   ├── text_processing.py  # Text processing operations
│   └── document_processing.py # Document processing operations
├── routes/                 # API routes
│   ├── __init__.py
│   ├── auth.py             # Authentication
│   ├── user_routes.py      # User-related routes
│   └── chatbot_routes.py   # Chatbot-related routes
└── chatbot/                # Chatbot functionality
    ├── __init__.py
    ├── model.py            # Model initialization
    ├── prompts.py          # Prompt templates
    └── rag.py              # RAG functionality
```

## Installation

### Install Ollama
You need to have Ollama installed before running:

```shell
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
```

### Setup Python Environment
Install Anaconda, then create and activate a new environment:

```shell
conda create -n genimagine_bot python=3.11 -y
conda activate genimagine_bot
pip install -r requirements.txt
```

You can edit qdrant-client to qdrant-client[fastembed-gpu] or vice versa, depending on the machine. Using qdrant-client is recommended when it is effective enough.

### Running the Application

```shell
python main.py
```

To kill the application if it's running on port 8000:
```shell
kill $(lsof -t -i:8000)
```

Alternative way to run with auto-reload:
```shell
uvicorn main:app --reload --port 9000
```

To expose the API using ngrok:
```shell
ngrok http http://localhost:8000
```

### Environment Variables

The application can be configured using the following environment variables:

- `OLLAMA_BASE_URL`: The base URL of the Ollama service (default: "http://localhost:11434")
- `MODEL_NAME`: The name of the Ollama model to use (default: "vinallama")
- `EMBEDDINGS_MODEL_PATH`: The path to the embeddings model (default: "./vietnamese-bi-encoder")
- `CUDA_VISIBLE_DEVICES`: The CUDA devices to use (default: "1")

Example:
```shell
OLLAMA_BASE_URL="http://ollama-server:11434" python main.py
```

## API Endpoints

### User Management
- `POST /register`: Register a new user
  - Parameters: `username` (string)

- `POST /update_business`: Update business text data
  - Parameters: `title` (string), `text` (string), `username` (string)

- `DELETE /delete_user`: Delete a user
  - Parameters: `username` (string)

- `DELETE /delete`: Delete user data
  - Parameters: `user_name` (string)

### Chatbot
- `GET /health/ollama`: Check if the Ollama service is running and accessible
  - Returns: Status of the Ollama service and available models

- `POST /add_qa_bot`: Add a QA pair for the bot
  - Parameters: `subject` (string), `question` (string), `answer` (string)

- `POST /ask_bot`: Ask a question to the bot
  - Parameters: `subject` (string), `username` (string), `question` (string)

- `POST /add_qa_for_business`: Add a QA pair for business
  - Parameters: `username` (string), `question` (string), `answer` (string)

- `POST /ask_business`: Ask a business-related question
  - Parameters: `username` (string), `question` (string)

