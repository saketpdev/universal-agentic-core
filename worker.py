import time
import logging
from dotenv import load_dotenv

from core.infrastructure import task_queue
from core.memory import session_manager
from core.engine.dag_runner import run_agentic_loop
from models.state import AgentRequest
from core.telemetry import TelemetryLogger
from models.telemetry import ActionStatus

# Setup Logging for the Worker Node
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("WorkerDaemon")
load_dotenv()

def start_worker():
    logger.info("🚀 Worker Node Booting Up... Listening to Redis Task Queue.")
    
    while True:
        try:
            # Blocks until a task arrives (0 = infinite wait)
            thread_id = task_queue.dequeue(timeout=0)
            
            if thread_id:
                # 1. INSTANT TELEMETRY: Stitch this process to the API Gateway's trace
                telemetry = TelemetryLogger(trace_id=thread_id)
                telemetry.log_decision(
                    agent_id="worker_daemon",
                    reasoning="Dequeued task from Redis. Initiating state hydration.",
                    context="Background Job Started"
                )
                
                logger.info(f"Worker picked up thread: {thread_id}")
                
                # TODO (Scale): If hitting DB race conditions under heavy concurrent load, 
                # add a time.sleep(0.05) here to let the API's SQLite commit finish syncing to disk.
                
                # 2. Hydrate state from DB
                briefcase = session_manager.get_briefcase(thread_id)
                if not briefcase:
                    error_msg = f"Critical Fault: Thread {thread_id} missing from SQLite DB! Dropping task."
                    logger.error(error_msg)
                    telemetry.log_decision(
                        agent_id="worker_daemon",
                        reasoning=error_msg,
                        context="State Hydration Failure"
                    )
                    continue
                
                request = AgentRequest(
                    user_prompt=briefcase.original_user_prompt,
                    user_id="system_worker_hydrated", 
                    thread_id=thread_id
                )
                
                # 3. Run the DAG
                start_time = time.time()
                response = run_agentic_loop(request)
                latency_ms = (time.time() - start_time) * 1000
                
                # 4. STRICT TERMINAL ROUTING (Database-Backed HRQ)
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

                    # Log the failure routing to Telemetry
                    telemetry.log_action(
                        agent_id="worker_daemon",
                        correlation_id=thread_id,
                        tool_name="hrq_routing",
                        arguments=f'{{"target_db": "human_reviews", "review_type": "{review_type}"}}',
                        status=ActionStatus.FAILED,
                        latency_ms=latency_ms,
                        result_summary=msg
                    )
                    
                    # Push to the Database-Backed Human Review Queue
                    session_manager.create_review_ticket(
                        thread_id=thread_id, 
                        review_type=review_type, 
                        message=msg
                    )
                    
                else:
                    logger.info(f"Worker successfully finished thread: {thread_id} in {latency_ms:.0f}ms")
                    
                    # Log the successful completion
                    telemetry.log_action(
                        agent_id="worker_daemon",
                        correlation_id=thread_id,
                        tool_name="dag_execution",
                        arguments='{"action": "run_agentic_loop"}',
                        status=ActionStatus.SUCCESS,
                        latency_ms=latency_ms,
                        result_summary=f"DAG Runner exited successfully with status: {response.status} in {response.iterations} iterations."
                    )
                
        except Exception as e:
            logger.error(f"Worker Loop Crashed: {str(e)}")
            time.sleep(5) # Circuit breaker to prevent rapid log-flooding

if __name__ == "__main__":
    start_worker()