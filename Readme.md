
# Gen Imagine RAG Client

Python client service for Gen Imagine RAG, providing a FastAPI backend that powers retrieval‑augmented chat with vector search (Qdrant) and local LLMs (via Ollama). This client complements the Gen Imagine RAG WebUI (Laravel) by exposing APIs to manage knowledge, optimize data/answers, and chat in two modes: general (Ask Bot) and collection‑scoped (Ask Business).

If you are looking for the WebUI repository, see: https://github.com/funnydevjsc/genimagine-rag-webui


## Key Concepts and Feature Mapping (Web ↔ Client)

In the WebUI, the term “subject” is used. In this Python client, the equivalent concept is “collection” (a Qdrant collection). Keep this mapping in mind:

- Ask Bot (Web = Ask Bot, Client = Ask Bot)
  - What it is: General chat across the full, global knowledge, not limited to a specific collection.
  - When to use: Broad questions or when you don’t know which collection is most relevant.

- Ask Business (Web = Ask Business, Client = Ask Business)
  - What it is: Chat scoped to a specific collection (called “subject” in the website; “collection” in the client). Retrieval only uses that collection’s vectors.
  - When to use: Deep dive into a particular business domain or dataset.

- Fine‑Tuning (Web = Fine Tuning, Client = Add QA Bot)
  - What it is: Add new question‑answer (QA) pairs to the global bot knowledge (global collection) so the bot can learn and retrieve from them.
  - Outcome: Data is embedded and stored in the vector database to improve general Ask Bot responses.

- Data‑Optimization (Web = Data Optimization, Client = Add QA Business)
  - What it is: Curate or add QA pairs into a specific collection to improve retrieval quality for that business domain.
  - Outcome: More precise and higher‑recall answers when using Ask Business for that collection.

- Response‑Optimization (Web = Response Optimization, Client = Response Optimization)
  - What it is: Techniques/prompts to improve how the model formulates answers based on curated QA pairs and retrieval context.
  - Outcome: More consistent, concise, and helpful responses.

- Conversations (Web = by subject, Client = by collection)
  - What it is: You can create and maintain multiple conversations organized by collection. Each conversation is a live chat session with the RAG bot, with history preserved to maintain context over turns.
  - Emphasis: “Subject” on the website equals “collection” in this client.


## What This Service Provides

- Vector search with Qdrant for high‑quality retrieval
- Local LLM inference via Ollama (configurable model)
- Two chat paths: Ask Bot (global) and Ask Business (collection‑scoped)
- APIs to add QA pairs globally (Add QA Bot) and per collection (Add QA Business)
- Conversation memory per user and category
- Utilities for loading and embedding documents/QA into Qdrant


## Project Structure (folders you will work with)

Top‑level directories/files:
- app/ – Main application code
  - config/ – Settings and runtime configuration (settings.py, GPU options)
  - models/ – Pydantic models for request/response payloads
  - database/ – DB integrations (MySQL helpers, Qdrant vector store helpers)
  - utils/ – Utilities for chat history, text/document processing
  - routes/ – FastAPI route handlers, including chatbot routes used by WebUI/client
  - chatbot/ – RAG core: model bootstrapping, prompts, and RAG pipeline
- main.py – FastAPI app entry point
- requirements.txt – Python dependencies
- environment.yaml – Conda environment spec
- collections.json – Example or seed collections configuration
- add_qa.py / add_knowledge.py / data_insert.py – Helper scripts to insert QA/document data
- chat_history/ – Stored conversation histories for users
- users/ – Local cache/storage for user‑related data
- response_cache/ – Cache for responses
- queues/ – Work queues (if used)
- qa_data_txt/ and qa_data_fixed.json – Example QA data sources
- docker_qdrant/ – Docker helpers for Qdrant setup
- vietnamese-bi-encoder/ – Embedding model assets (local path used by default)

Within app/ important files include:
- app/chatbot/rag.py – Implements the RAG answer flows:
  - answer_business(subject, question, user_id) → collection‑scoped retrieval
  - answer_user(question, user_id) → global retrieval
- app/routes/chatbot_routes.py – API endpoints for asking and adding QA
- app/database/vector_db.py – Creating collections and adding/searching documents
- app/utils/chat_history.py – Persist and load conversation history per user/category


## Installation

1) Install Ollama (for local LLM inference):

shell
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2

2) Create and activate a Python environment:

shell
conda create -n genimagine_bot python=3.11 -y
conda activate genimagine_bot
pip install -r requirements.txt

Note: You can switch between qdrant-client and qdrant-client[fastembed-gpu] depending on your machine. Start with qdrant-client unless you need GPU acceleration for embeddings.

3) Run the service:

shell
python main.py

Useful commands:
- Kill a process running on port 8000:
  shell
  kill $(lsof -t -i:8000)
- Run with auto‑reload on port 9000:
  shell
  uvicorn main:app --reload --port 9000
- Expose the API with ngrok:
  shell
  ngrok http http://localhost:8000


## Configuration (environment variables)

- OLLAMA_BASE_URL – Base URL of your Ollama instance (default: http://localhost:11434)
- MODEL_NAME – Ollama model name (default: vinallama)
- EMBEDDINGS_MODEL_PATH – Path to local embeddings model (default: ./vietnamese-bi-encoder)
- CUDA_VISIBLE_DEVICES – CUDA device selection (default: 1)

Example:

shell
OLLAMA_BASE_URL="http://ollama-server:11434" python main.py


## Sample knowledge base

This repository references a sample knowledge base you can use to build a Qdrant vector database:
https://drive.google.com/file/d/1BS3NK_m3BMDig5ZNLp27mA7wuPq1Ceun/view?usp=drive_link


## API Endpoints (aligned to concepts)

User Management
- POST /register – Register a new user
  - Body: username
- POST /update_business – Update business text data
  - Body: title, text, username
- DELETE /delete_user – Delete a user
  - Body: username
- DELETE /delete – Delete all user data
  - Body: user_name

Health
- GET /health/ollama – Check Ollama status and list models

Knowledge Management
- POST /add_qa_bot – Fine‑Tuning (Client: Add QA Bot, Web: Fine Tuning)
  - Adds a QA pair to the global knowledge (affects Ask Bot)
  - Body: subject, question, answer
    - Note: subject here can be used to group under a collection name if desired; global usage is supported by implementation.
- POST /add_qa_for_business – Data‑Optimization (Client: Add QA Business, Web: Data Optimization)
  - Adds a QA pair to a specific collection to optimize domain retrieval
  - Body: username, question, answer

Chat
- POST /ask_bot – Ask Bot (general)
  - Body: subject, username, question
  - Behavior: Not limited by a specific collection; uses global knowledge
- POST /ask_business – Ask Business (collection‑scoped)
  - Body: username, question
  - Behavior: Answers based on a specific collection (“subject” in WebUI; “collection” in client). The collection context is tied to the user/session per implementation.

Notes:
- Conversations: The service stores per‑user conversation history (by category/collection) to maintain multi‑turn context.
- Response‑Optimization happens inside the RAG and prompt composition layers (see app/chatbot/prompts.py and app/chatbot/rag.py).


## Usage Examples

Ask Bot (general):

shell
curl -X POST http://localhost:8000/ask_bot \
  -H "Content-Type: application/json" \
  -d '{"subject":"general","username":"alice","question":"What is Gen Imagine RAG?"}'

Ask Business (by collection):

shell
curl -X POST http://localhost:8000/ask_business \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","question":"How do I onboard a new merchant?"}'

Add QA Bot (Fine‑Tuning, global):

shell
curl -X POST http://localhost:8000/add_qa_bot \
  -H "Content-Type: application/json" \
  -d '{"subject":"global","question":"Q here","answer":"A here"}'

Add QA Business (Data‑Optimization, per collection):

shell
curl -X POST http://localhost:8000/add_qa_for_business \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","question":"Q here","answer":"A here"}'


## How it works (high level)

- The client embeds documents/QA pairs and stores them in Qdrant collections (vector_db.py).
- For each user query, the service retrieves top‑k vectors from the relevant collection(s) and constructs a context.
- The context plus user question is fed to an Ollama‑served LLM using carefully crafted prompts (prompts.py) to produce the final answer.
- Conversation history is persisted to improve continuity over multiple turns.


## Feedback

Respect us in the [Laravel Việt Nam](https://www.facebook.com/groups/167363136987053)

## Contributing

Please see [CONTRIBUTING](CONTRIBUTING.md) for details.

### Security

If you discover any security related issues, please email contact@funnydev.vn or use the issue tracker.

## Credits

- [Funny Dev., Jsc](https://github.com/funnydevjsc)
- [All Contributors](../../contributors)

## License

The MIT License (MIT). Please see [License File](LICENSE.md) for more information.
