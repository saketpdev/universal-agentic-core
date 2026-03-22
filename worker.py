import time
import logging
import asyncio
from dotenv import load_dotenv

from core.mcp.manager import mcp_manager
from core.infrastructure import task_queue
from core.memory import session_manager
from core.engine.dag_runner import run_agentic_loop
from core.telemetry import TelemetryLogger

from models.state import AgentRequest
from models.telemetry import ActionStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("WorkerDaemon")
load_dotenv()

MAX_CONCURRENT_TASKS = 10
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

async def process_workflow(thread_id: str):
    """Handles the actual execution of a single user's DAG."""
    async with semaphore:
        try:
            telemetry = TelemetryLogger(trace_id=thread_id)
            await telemetry.log_decision(
                agent_id="worker_daemon",
                reasoning="Dequeued task from Redis. Initiating state hydration.",
                context="Background Job Started"
            )

            logger.info(f"Worker processing thread: {thread_id} (Active Slots: {MAX_CONCURRENT_TASKS - semaphore._value}/{MAX_CONCURRENT_TASKS})")

            # 🚀 NON-BLOCKING SQLITE FETCH
            briefcase = await asyncio.to_thread(session_manager.get_briefcase, thread_id)

            if not briefcase:
                error_msg = f"Critical Fault: Thread {thread_id} missing from SQLite DB! Dropping task."
                logger.error(error_msg)
                await telemetry.log_decision("worker_daemon", error_msg, "State Hydration Failure")
                return

            request = AgentRequest(
                user_prompt=briefcase.original_user_prompt,
                user_id="system_worker_hydrated",
                thread_id=thread_id
            )

            start_time = time.time()
            response = await run_agentic_loop(request)
            latency_ms = (time.time() - start_time) * 1000

            # --- TERMINAL ROUTING ---
            if response.status in ["budget_exceeded", "error", "security_violation"]:
                review_type = "UNKNOWN"
                msg = "Workflow halted for unknown reasons."

                if response.status == "budget_exceeded":
                    review_type = "FINOPS"
                    msg = "Workflow halted due to empty FinOps ledger. Requires fund top-up."
                    logger.error(f"FINOPS TERMINAL: {thread_id} ran out of money.")
                elif response.status == "error":
                    review_type = "LOGIC"
                    msg = f"Terminal LLM logic failure at iteration {response.iterations}. Requires prompt inspection."
                    logger.error(f"LOGIC TERMINAL: {thread_id} crashed natively.")
                elif response.status == "security_violation":
                    review_type = "SECURITY"
                    msg = "Security Action Gate triggered. Requires human manager approval."
                    logger.critical(f"SECURITY TERMINAL: {thread_id} attempted irreversible action.")

                await telemetry.log_action(
                    agent_id="worker_daemon",
                    correlation_id=thread_id,
                    tool_name="hrq_routing",
                    arguments=f'{{"target_db": "human_reviews", "review_type": "{review_type}"}}',
                    status=ActionStatus.FAILED,
                    latency_ms=latency_ms,
                    result_summary=msg
                )

                await asyncio.to_thread(session_manager.create_review_ticket, thread_id, review_type, msg)

            else:
                logger.info(f"✅ Worker finished thread: {thread_id} in {latency_ms:.0f}ms")
                await telemetry.log_action(
                    agent_id="worker_daemon",
                    correlation_id=thread_id,
                    tool_name="dag_execution",
                    arguments='{"action": "run_agentic_loop"}',
                    status=ActionStatus.SUCCESS,
                    latency_ms=latency_ms,
                    result_summary=f"DAG exited with status: {response.status} in {response.iterations} iterations."
                )

        except Exception as e:
            logger.error(f"Thread {thread_id} Crashed Hard: {str(e)}")

async def start_worker():
    """The Infinite Event Loop."""

    # Boot up all external MCP network connections!
    logger.info("Initializing MCP Network Connections...")
    await mcp_manager.connect_all()

    logger.info(f"🚀 Async Worker Node Booting Up... (Max Concurrency: {MAX_CONCURRENT_TASKS})")

    while True:
        try:
            thread_id = await task_queue.dequeue(timeout=1)

            if thread_id:
                asyncio.create_task(process_workflow(thread_id))

        except Exception as e:
            logger.error(f"Redis Polling Crashed: {str(e)}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(start_worker())
    except KeyboardInterrupt:
        logger.info("\nShutdown signal received. Closing event loop...")
    finally:
        logger.info("Gracefully severing all MCP Network connections...")
        asyncio.run(mcp_manager.disconnect_all())
        logger.info("Worker Node successfully terminated.")