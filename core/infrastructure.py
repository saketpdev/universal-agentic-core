import redis
import uuid
import logging
import hashlib

logger = logging.getLogger("AgenticCore.Infrastructure")
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

class RedisBudgetManager:
    def reserve(self, amount: float, ttl: int) -> str:
        lock_id = f"budget_lock_{uuid.uuid4().hex[:8]}"
        redis_client.set(lock_id, str(amount), ex=ttl)
        return lock_id

    def release(self, lock_id: str):
        redis_client.delete(lock_id)
        logger.info(f"Budget lock {lock_id} released.")

class RedisIdempotencyRegistry:
    def generate_hash(self, thread_id: str, func_name: str, args_str: str) -> str:
        """Creates a deterministic, session-scoped hash for the exact tool call."""
        raw_string = f"{thread_id}:{func_name}:{args_str}"
        return hashlib.md5(raw_string.encode()).hexdigest()

    def check_idempotency(self, hash_key: str) -> bool:
        return redis_client.exists(f"idem:{hash_key}") > 0

    def get_result(self, hash_key: str) -> str:
        return redis_client.get(f"idem:{hash_key}")

    def save_idempotency(self, hash_key: str, result: str):
        redis_client.set(f"idem:{hash_key}", result, ex=86400) # 24 hour cache

budget_manager = RedisBudgetManager()
db = RedisIdempotencyRegistry()