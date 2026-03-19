# Universal Agentic Core

An enterprise-grade, asynchronous, multi-agent AI backend. This system features a DAG-based orchestration engine, dynamic Swarm handoffs, FinOps circuit breakers, an OpenTelemetry-instrumented worker queue, and persistent episodic memory backed by SQLite and Redis.

## 🏗️ Architecture Overview

The system is decoupled into three primary layers for horizontal scalability:

1. **API Gateway (`main.py` & `api/routes.py`):** A lightweight FastAPI server that receives user prompts, initializes state in the SQLite Database, pushes tasks to Redis, and immediately returns a `202 Accepted` response to prevent HTTP timeouts.
2. **Message Bus (Redis):** Handles asynchronous task queuing (`agentic:task_queue`).
3. **Worker Daemon (`worker.py`):** An infinitely looping background process that pulls tasks from Redis, executes the LLM reasoning chain (ReAct), handles dynamic agent handoffs, and manages terminal failures (routing to the Human-in-the-Loop Database).

## ⚙️ Prerequisites

- Python 3.10+
- Local Redis Server (or Dockerized Redis)
- Required API Keys (Groq, OpenAI, or Anthropic depending on your registry config)

## 🛠️ Installation

1. **Clone the repository and enter the directory:**

   ```bash
   git clone <your-repo-url>
   cd universal-agentic-core
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your provider keys:
   ```env
   GROQ_API_KEY=your_groq_key_here
   # Add other keys as configured in core/model_registry.py
   ```

## 🚀 Running the Cluster (Local Development)

To run the full distributed architecture, you will need three separate terminal windows.

**Terminal 1: The Message Bus**
Start your local Redis server.

```bash
redis-server
```

**Terminal 2: The API Server**
Start the FastAPI gateway. This will listen for incoming HTTP requests on port 8000.

```bash
python main.py
```

**Terminal 3: The Background Worker**
Start the worker daemon. You can run multiple instances of this script across different terminals to process tasks concurrently.

```bash
python worker.py
```

## 📡 Usage

Send a POST request to the API to trigger a workflow. The API will respond instantly with a `202 Accepted` style payload, and the background worker will begin processing.

**cURL Request:**

```bash
curl -X POST http://localhost:8000/execute \
-H "Content-Type: application/json" \
-d '{
  "user_id": "usr_123",
  "user_prompt": "Audit my invoice for mathematical errors."
}'
```

**Expected API Response:**

```json
{
  "status": "accepted",
  "thread_id": "a1b2c3d4-5678-90ef-ghij-klmnopqrstuv",
  "message": "Workflow queued successfully. Poll the database for updates."
}
```

Watch **Terminal 3** to see the worker pick up the task, execute the DAG, log traces to OpenTelemetry, and save the final resolution!
