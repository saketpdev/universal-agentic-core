import json
import time
from typing import Any, Optional, cast
import redis.asyncio as redis  # 🚀 SWITCHED TO ASYNC REDIS
import uuid
import logging
import hashlib
from opentelemetry import baggage

logger = logging.getLogger("AgenticCore.Infrastructure")
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

class BudgetExceededException(Exception):
    pass

class FinOpsLedger:
    def __init__(self, redis_conn):
        self.redis = redis_conn

    def _get_current_tenant(self) -> str:
        tenant_id = baggage.get_baggage("tenant_id")
        if not tenant_id:
            return "system_default"
        return str(tenant_id)

    async def get_balance(self) -> float:
        tenant_id = self._get_current_tenant()
        val = await self.redis.get(f"ledger:{tenant_id}:balance")
        return float(val) if val else 0.0

    async def add_funds(self, tenant_id: str, amount_usd: float):
        await self.redis.incrbyfloat(f"ledger:{tenant_id}:balance", amount_usd)
        logger.info(f"FinOps: Added ${amount_usd:.2f} to tenant {tenant_id}")

    async def reserve_deposit(self, trace_id: str, max_budget_usd: float):
        tenant_id = self._get_current_tenant()
        current_balance = await self.get_balance()

        if current_balance < max_budget_usd:
            raise BudgetExceededException(
                f"FinOps Rejected: Tenant {tenant_id} balance (${current_balance:.2f}) "
                f"is below the required workflow deposit (${max_budget_usd:.2f})."
            )

        await self.redis.incrbyfloat(f"ledger:{tenant_id}:balance", -max_budget_usd)
        await self.redis.set(f"escrow:{trace_id}", max_budget_usd)
        logger.info(f"FinOps: Reserved ${max_budget_usd:.2f} deposit for trace {trace_id}")

    async def burn_down(self, trace_id: str, cost_usd: float):
        if cost_usd <= 0:
            return

        tenant_id = self._get_current_tenant()
        remaining = await self.redis.incrbyfloat(f"escrow:{trace_id}", -cost_usd)

        if remaining <= 0:
            logger.error(f"FinOps: CIRCUIT BREAKER TRIPPED for tenant {tenant_id} on trace {trace_id}!")
            raise BudgetExceededException("Hard token budget exceeded. Task forcefully terminated.")

    async def release_deposit(self, trace_id: str):
        tenant_id = self._get_current_tenant()
        escrow_val = await self.redis.get(f"escrow:{trace_id}")

        if escrow_val:
            remaining = float(escrow_val)
            if remaining > 0:
                await self.redis.incrbyfloat(f"ledger:{tenant_id}:balance", remaining)
                logger.info(f"FinOps: Workflow complete. Refunded ${remaining:.5f} to {tenant_id}")

            await self.redis.delete(f"escrow:{trace_id}")


class RedisIdempotencyRegistry:
    def generate_hash(self, thread_id: str, func_name: str, args_str: str) -> str:
        raw_string = f"{thread_id}:{func_name}:{args_str}"
        return hashlib.md5(raw_string.encode()).hexdigest()

    async def check_idempotency(self, hash_key: str) -> bool:
        int_val = int(cast(Any, await redis_client.exists(f"idem:{hash_key}")) or 0)
        return  int_val > 0

    async def get_result(self, hash_key: str) -> Optional[str]:
        result = await redis_client.get(f"idem:{hash_key}")
        if result is None:
            return None
        return result.decode("utf-8") if isinstance(result, bytes) else str(result)

    async def save_idempotency(self, hash_key: str, result: str):
        await redis_client.set(f"idem:{hash_key}", result, ex=86400) 

class RedisTaskQueue:
    def __init__(self, redis_conn):
        self.redis = redis_conn
        self.queue_name = "agentic:task_queue"

    async def enqueue(self, thread_id: str):
        await self.redis.lpush(self.queue_name, thread_id)
        logger.info(f"Queue: Enqueued thread {thread_id} for background processing.")

    async def dequeue(self, timeout: int = 0) -> Optional[str]:
        result = await self.redis.brpop(self.queue_name, timeout=timeout)
        if result:
            return result[1]
        return None

budget_manager = FinOpsLedger(redis_client)
task_queue = RedisTaskQueue(redis_client)
db = RedisIdempotencyRegistry()