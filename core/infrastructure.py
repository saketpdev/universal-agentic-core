import json
import time
from typing import Any, Optional, cast
import redis
import uuid
import logging
import hashlib
from opentelemetry import baggage

logger = logging.getLogger("AgenticCore.Infrastructure")
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

class BudgetExceededException(Exception):
    """Raised when an agent burns through its pre-flight deposit."""
    pass

class FinOpsLedger:
    def __init__(self, redis_conn):
        self.redis = redis_conn

    def _get_current_tenant(self) -> str:
        """Securely extracts the opaque tenant_id from OpenTelemetry Baggage."""
        tenant_id = baggage.get_baggage("tenant_id")
        if not tenant_id:
            logger.warning("FinOps: No tenant_id found in Baggage. Defaulting to 'system_default'.")
            return "system_default"
        return str(tenant_id)

    def get_balance(self) -> float:
        """Check the tenant's current total balance."""
        tenant_id = self._get_current_tenant()
        val = self.redis.get(f"ledger:{tenant_id}:balance")
        return float(val) if val else 0.0

    def add_funds(self, tenant_id: str, amount_usd: float):
        """Admin function to credit a tenant's account."""
        self.redis.incrbyfloat(f"ledger:{tenant_id}:balance", amount_usd)
        logger.info(f"FinOps: Added ${amount_usd:.2f} to tenant {tenant_id}")

    def reserve_deposit(self, trace_id: str, max_budget_usd: float):
        """The Pre-Flight Check. Reserves funds before the DAG starts."""
        tenant_id = self._get_current_tenant()
        current_balance = self.get_balance()

        if current_balance < max_budget_usd:
            raise BudgetExceededException(
                f"FinOps Rejected: Tenant {tenant_id} balance (${current_balance:.2f}) "
                f"is below the required workflow deposit (${max_budget_usd:.2f})."
            )

        self.redis.incrbyfloat(f"ledger:{tenant_id}:balance", -max_budget_usd)
        self.redis.set(f"escrow:{trace_id}", max_budget_usd)
        logger.info(f"FinOps: Reserved ${max_budget_usd:.2f} deposit for trace {trace_id}")

    def burn_down(self, trace_id: str, cost_usd: float):
        """The Circuit Breaker. Deducts micro-costs from the escrow."""
        if cost_usd <= 0:
            return

        tenant_id = self._get_current_tenant()
        remaining = self.redis.incrbyfloat(f"escrow:{trace_id}", -cost_usd)

        # 💥 THE HARD CIRCUIT BREAKER 💥
        if remaining <= 0:
            logger.error(f"FinOps: CIRCUIT BREAKER TRIPPED for tenant {tenant_id} on trace {trace_id}!")
            raise BudgetExceededException("Hard token budget exceeded. Task forcefully terminated.")

    def release_deposit(self, trace_id: str):
        """The Reconciliation. Refunds unspent escrow back to the ledger."""
        tenant_id = self._get_current_tenant()
        escrow_val = self.redis.get(f"escrow:{trace_id}")

        if escrow_val:
            remaining = float(escrow_val)
            if remaining > 0:
                self.redis.incrbyfloat(f"ledger:{tenant_id}:balance", remaining)
                logger.info(f"FinOps: Workflow complete. Refunded ${remaining:.5f} to {tenant_id}")

            self.redis.delete(f"escrow:{trace_id}")


class RedisIdempotencyRegistry:
    def generate_hash(self, thread_id: str, func_name: str, args_str: str) -> str:
        """Creates a deterministic, session-scoped hash for the exact tool call."""
        raw_string = f"{thread_id}:{func_name}:{args_str}"
        return hashlib.md5(raw_string.encode()).hexdigest()

    def check_idempotency(self, hash_key: str) -> bool:

        int_val = int(cast(Any, redis_client.exists(f"idem:{hash_key}")) or 0)
        return  int_val > 0

    def get_result(self, hash_key: str)-> Optional[str]:
        result = redis_client.get(f"idem:{hash_key}")
        if result is None:
            return None
        # If it's bytes, decode it
        return result.decode("utf-8") if isinstance(result, bytes) else str(result)

    def save_idempotency(self, hash_key: str, result: str):
        redis_client.set(f"idem:{hash_key}", result, ex=86400) # 24 hour cache

class RedisTaskQueue:
    def __init__(self, redis_conn):
        self.redis = redis_conn
        self.queue_name = "agentic:task_queue"

    def enqueue(self, thread_id: str):
        """Pushes a workflow thread ID to the back of the queue."""
        self.redis.lpush(self.queue_name, thread_id)
        logger.info(f"Queue: Enqueued thread {thread_id} for background processing.")

    def dequeue(self, timeout: int = 0) -> Optional[str]:
        """
        Blocks and waits for a new task. 
        Returns the thread_id, or None if timeout is reached.
        """
        result = self.redis.brpop(self.queue_name, timeout=timeout)
        if result:
            # result is a tuple: (queue_name, value)
            return result[1]
        return None

# Initializing instances
budget_manager = FinOpsLedger(redis_client)
task_queue = RedisTaskQueue(redis_client)

db = RedisIdempotencyRegistry()