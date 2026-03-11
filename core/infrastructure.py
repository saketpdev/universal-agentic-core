import redis
import uuid
import logging

logger = logging.getLogger("AgenticCore.Infrastructure")
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

class RedisBudgetManager:
    def reserve(self, amount: float, ttl: int) -> str:
        lock_id = f"budget_lock_{uuid.uuid4().hex[:8]}"
        redis_client.set(lock_id, str(amount), ex=ttl)
        return lock_id

    def release(self, lock_id: str):
        redis_client.delete(lock_id)
        logger.info(f"Budget lock {lock_id} released in Redis.")

class RedisIdempotencyRegistry:
    def check_idempotency(self, call_id: str) -> bool:
        return redis_client.exists(f"idem:{call_id}") > 0

    def get_result(self, call_id: str) -> str:
        return redis_client.get(f"idem:{call_id}")

    def save_idempotency(self, call_id: str, result: str):
        redis_client.set(f"idem:{call_id}", result, ex=86400) # 24 hour cache

budget_manager = RedisBudgetManager()
db = RedisIdempotencyRegistry()