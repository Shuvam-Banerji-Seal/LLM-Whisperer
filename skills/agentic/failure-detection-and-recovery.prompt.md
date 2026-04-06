# Failure Detection and Recovery for Agents Skill

**Author:** Shuvam Banerji Seal  
**Version:** 1.0  
**Last Updated:** 2026-04-06

## Identity and Mission

This skill provides comprehensive failure detection and recovery patterns for distributed agent systems. It covers health checks, heartbeats, circuit breakers, retry strategies, graceful degradation, and auto-recovery mechanisms essential for building resilient multi-agent orchestrations.

## Problem Definition

Distributed failures are inevitable and must be detected and recovered from:

1. **Transient Failures:** Network hiccups, temporary overload (recoverable with retry)
2. **Permanent Failures:** Hardware crash, persistent service unavailability (need failover)
3. **Cascading Failures:** One agent's failure triggers others (circuit breaker prevention)
4. **Slow Failures:** Service responding but very slowly (timeout detection)
5. **Silent Failures:** Component failing but no error signal (heartbeat detection)
6. **Recovery Strategies:** Retry, failover, graceful degradation, circuit breaking

## Architecture Patterns

### Health Check and Heartbeat

```
ACTIVE HEALTH CHECKS
====================

Healthy Agent:
Monitor -> HTTP GET /health -> 200 OK
           |<------ 50ms ------>|
           
Every 5 seconds. If 3 consecutive checks fail -> UNHEALTHY


PASSIVE HEARTBEAT MONITORING
=============================

Agent sends heartbeat every 10 seconds:
Leader <- Heartbeat <- Agent1
Leader <- Heartbeat <- Agent2
Leader <- Heartbeat <- Agent3

Leader timeout = 30 seconds (3x heartbeat interval)
If no heartbeat in 30 seconds -> DEAD, trigger failover


LIVENESS vs READINESS
=====================

Liveness: Is service alive? (process running, not deadlocked)
  - If false: Container should restart
  - Health check: Can service accept requests?

Readiness: Can service handle traffic now?
  - If false: Service alive but not ready (loading cache, warming up)
  - Health check: Startup tasks complete? Dependencies available?

Example:
  Liveness: Always true unless deadlocked
  Readiness: False during 5-min startup, true after initialization
```

### Circuit Breaker Pattern

```
CIRCUIT BREAKER STATE MACHINE
==============================

                 [CLOSED] (normal state)
                 /      \
            success      failure_threshold_exceeded
            /                    \
          /                        \
       [CLOSED]                 [OPEN]
       Normal ops              Fail fast
                                  |
                           (timeout: 60s)
                                  |
                              [HALF-OPEN]
                               Try 1 req
                                  |
                         +--------+--------+
                         |                 |
                      success           failure
                         |                 |
                       [CLOSED]          [OPEN]
                     Resume ops         Back to failing


EXAMPLE: Calling failing service
=================================

Attempt 1-5: Call service
  - Service timeout -> Increment failure_count
  - failure_count >= threshold (5) -> OPEN

Circuit now OPEN:
  - Attempt 6-60: Immediately fail (no network call)
  - Fast failure, prevent cascading
  
After 60 seconds:
  - HALF-OPEN state
  - Try single request
  - If succeeds: CLOSED (resume normal)
  - If fails: OPEN (wait another 60s)
```

### Retry Strategies

```
EXPONENTIAL BACKOFF WITH JITTER
===============================

Attempt 1: immediate
Attempt 2: wait 100ms + jitter
Attempt 3: wait 200ms + jitter
Attempt 4: wait 400ms + jitter
Attempt 5: wait 800ms + jitter

Jitter: random(0, backoff_time)
Purpose: Prevent thundering herd (all clients retry simultaneously)

Formula: wait_time = min(base_delay * 2^attempt, max_delay) + random(0, jitter)

Example with max_delay=10000ms:
Attempt 1: 0ms
Attempt 2: 100 + 50 = 150ms
Attempt 3: 200 + 75 = 275ms
Attempt 4: 400 + 200 = 600ms
Attempt 5: 800 + 400 = 1200ms


IDEMPOTENCY FOR SAFE RETRIES
=============================

Idempotent operations: Can retry without harmful side effects
  - GET request (read-only)
  - PUT with same body (idempotent write)
  - DELETE (idempotent)

Non-idempotent:
  - POST (may create duplicate resource)
  - Financial transactions

Safe retry: Only retry IDEMPOTENT operations
Use idempotency token to make POST safe:
  POST /order with Idempotency-Key: abc123
  Multiple requests with same key -> same result
```

## Python Implementation - Failure Detection and Recovery

```python
"""
Production failure detection and recovery system.
Includes health checks, circuit breakers, smart retries, and auto-recovery.
"""

import asyncio
import time
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Awaitable, TypeVar
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    healthy: bool
    status_code: Optional[int] = None
    response_time_ms: float = 0.0
    message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class HealthChecker:
    """
    Monitors agent health via periodic checks.
    Detects both liveness (is it running?) and readiness (can it handle traffic?).
    """
    
    def __init__(
        self,
        agent_id: str,
        check_interval_seconds: int = 5,
        unhealthy_threshold: int = 3,
        timeout_seconds: int = 2
    ):
        self.agent_id = agent_id
        self.check_interval_seconds = check_interval_seconds
        self.unhealthy_threshold = unhealthy_threshold
        self.timeout_seconds = timeout_seconds
        
        self.is_healthy = True
        self.consecutive_failures = 0
        self.check_history: deque = deque(maxlen=100)
        self.last_check_time: Optional[datetime] = None
    
    async def check_liveness(self) -> HealthCheckResult:
        """Check if agent process is alive."""
        try:
            # Simulate health endpoint call
            start = time.time()
            await asyncio.sleep(random.uniform(0.01, 0.05))
            response_time_ms = (time.time() - start) * 1000
            
            # Random 95% success rate
            if random.random() < 0.95:
                return HealthCheckResult(
                    healthy=True,
                    status_code=200,
                    response_time_ms=response_time_ms,
                    message="Liveness check passed"
                )
            else:
                return HealthCheckResult(
                    healthy=False,
                    status_code=500,
                    response_time_ms=response_time_ms,
                    message="Liveness check failed"
                )
        
        except asyncio.TimeoutError:
            return HealthCheckResult(
                healthy=False,
                status_code=None,
                response_time_ms=self.timeout_seconds * 1000,
                message="Liveness check timeout"
            )
    
    async def check_readiness(self) -> HealthCheckResult:
        """Check if agent can handle traffic."""
        try:
            # Simulate readiness endpoint
            start = time.time()
            await asyncio.sleep(random.uniform(0.01, 0.03))
            response_time_ms = (time.time() - start) * 1000
            
            # Higher success rate (readiness more stable than liveness)
            if random.random() < 0.98:
                return HealthCheckResult(
                    healthy=True,
                    status_code=200,
                    response_time_ms=response_time_ms,
                    message="Readiness check passed"
                )
            else:
                return HealthCheckResult(
                    healthy=False,
                    status_code=503,
                    response_time_ms=response_time_ms,
                    message="Not ready (dependencies unavailable)"
                )
        
        except asyncio.TimeoutError:
            return HealthCheckResult(
                healthy=False,
                status_code=None,
                response_time_ms=self.timeout_seconds * 1000,
                message="Readiness check timeout"
            )
    
    async def run_health_checks(self) -> None:
        """Continuously run health checks."""
        while True:
            liveness = await self.check_liveness()
            readiness = await self.check_readiness()
            
            is_healthy = liveness.healthy and readiness.healthy
            self.check_history.append({
                "timestamp": datetime.now(),
                "liveness": liveness.healthy,
                "readiness": readiness.healthy,
                "liveness_time_ms": liveness.response_time_ms,
                "readiness_time_ms": readiness.response_time_ms,
            })
            
            if not is_healthy:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.unhealthy_threshold:
                    self.is_healthy = False
                    logger.error(f"Agent {self.agent_id} marked UNHEALTHY "
                               f"({self.consecutive_failures} consecutive failures)")
            else:
                if self.consecutive_failures > 0:
                    logger.info(f"Agent {self.agent_id} health recovered")
                self.consecutive_failures = 0
                self.is_healthy = True
            
            self.last_check_time = datetime.now()
            await asyncio.sleep(self.check_interval_seconds)
    
    def get_status(self) -> Dict[str, Any]:
        """Get health checker status."""
        return {
            "agent_id": self.agent_id,
            "is_healthy": self.is_healthy,
            "consecutive_failures": self.consecutive_failures,
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
            "recent_checks": list(self.check_history)[-10:]
        }


class CircuitBreaker:
    """
    Prevents cascading failures by failing fast when downstream is unavailable.
    Implements state machine: CLOSED -> OPEN -> HALF_OPEN -> CLOSED
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_seconds: int = 60,
        expected_exception: type = Exception
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.expected_exception = expected_exception
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state_change_time = datetime.now()
        
        self.call_history: deque = deque(maxlen=100)
    
    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker.
        
        Raises:
            Exception if circuit is OPEN (fail fast)
            Original exception if function fails in CLOSED/HALF_OPEN
        """
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logger.info(f"CircuitBreaker {self.name}: Attempting reset (HALF_OPEN)")
            else:
                logger.debug(f"CircuitBreaker {self.name}: Circuit OPEN, failing fast")
                raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            
            self._on_success()
            self.call_history.append({
                "timestamp": datetime.now(),
                "state": self.state.value,
                "success": True
            })
            
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            self.call_history.append({
                "timestamp": datetime.now(),
                "state": self.state.value,
                "success": False,
                "error": str(e)
            })
            raise
    
    def _on_success(self) -> None:
        """Handle successful call."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 1:  # Single success closes circuit
                self._reset()
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        logger.warning(f"CircuitBreaker {self.name}: Failure {self.failure_count}/{self.failure_threshold}")
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.state_change_time = datetime.now()
            logger.error(f"CircuitBreaker {self.name}: Circuit OPEN (too many failures)")
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.state != CircuitBreakerState.OPEN:
            return False
        
        time_in_open = (datetime.now() - self.state_change_time).total_seconds()
        return time_in_open >= self.recovery_timeout_seconds
    
    def _reset(self) -> None:
        """Reset circuit to CLOSED."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.state_change_time = datetime.now()
        logger.info(f"CircuitBreaker {self.name}: Circuit CLOSED (recovered)")
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "time_in_state_seconds": (datetime.now() - self.state_change_time).total_seconds(),
            "recent_calls": list(self.call_history)[-20:]
        }


class SmartRetryPolicy:
    """
    Implements exponential backoff with jitter and idempotency awareness.
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay_ms: int = 100,
        max_delay_ms: int = 10000,
        jitter_factor: float = 0.1,
        idempotent: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay_ms = base_delay_ms
        self.max_delay_ms = max_delay_ms
        self.jitter_factor = jitter_factor
        self.idempotent = idempotent
    
    async def execute_with_retries(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs
    ) -> T:
        """Execute function with retry logic."""
        last_error = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                logger.debug(f"Attempt {attempt}/{self.max_attempts}")
                return await func(*args, **kwargs)
            
            except Exception as e:
                last_error = e
                
                # Don't retry non-idempotent operations
                if not self.idempotent:
                    logger.error(f"Non-idempotent operation failed, not retrying: {e}")
                    raise
                
                if attempt < self.max_attempts:
                    wait_time_ms = self._calculate_backoff(attempt)
                    logger.warning(f"Attempt {attempt} failed: {e}. "
                                 f"Retrying in {wait_time_ms}ms...")
                    await asyncio.sleep(wait_time_ms / 1000)
        
        logger.error(f"All {self.max_attempts} attempts failed: {last_error}")
        raise last_error
    
    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter."""
        exponential_delay = self.base_delay_ms * (2 ** (attempt - 1))
        capped_delay = min(exponential_delay, self.max_delay_ms)
        jitter = random.uniform(0, capped_delay * self.jitter_factor)
        return capped_delay + jitter
    
    def get_config(self) -> Dict[str, Any]:
        """Get retry policy configuration."""
        return {
            "max_attempts": self.max_attempts,
            "base_delay_ms": self.base_delay_ms,
            "max_delay_ms": self.max_delay_ms,
            "jitter_factor": self.jitter_factor,
            "idempotent": self.idempotent,
        }


class ResilientAgent:
    """
    Combines health checking, circuit breaking, and retry logic for resilient operations.
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.health_checker = HealthChecker(agent_id)
        self.circuit_breaker = CircuitBreaker(f"cb_{agent_id}")
        self.retry_policy = SmartRetryPolicy(max_attempts=3, idempotent=True)
    
    async def execute_resilient_operation(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs
    ) -> T:
        """Execute operation with full resilience stack."""
        
        # Check health first
        if not self.health_checker.is_healthy:
            raise Exception(f"Agent {self.agent_id} is unhealthy")
        
        # Try through circuit breaker with retries
        async def circuit_breaker_call():
            return await self.circuit_breaker.call(func, *args, **kwargs)
        
        return await self.retry_policy.execute_with_retries(circuit_breaker_call)
    
    def get_resilience_status(self) -> Dict[str, Any]:
        """Get full resilience status."""
        return {
            "agent_id": self.agent_id,
            "health": self.health_checker.get_status(),
            "circuit_breaker": self.circuit_breaker.get_status(),
            "retry_policy": self.retry_policy.get_config(),
        }


# ============ EXAMPLE USAGE ============

async def example_failure_recovery():
    """Example: Resilient operation with all recovery mechanisms."""
    
    agent = ResilientAgent("critical_agent")
    
    # Start health checks
    health_task = asyncio.create_task(agent.health_checker.run_health_checks())
    
    async def failing_operation():
        """Simulates operation that may fail."""
        if random.random() < 0.3:
            raise Exception("Transient failure")
        return {"status": "success", "data": "important"}
    
    # Attempt resilient execution multiple times
    successes = 0
    failures = 0
    
    for i in range(10):
        try:
            result = await agent.execute_resilient_operation(failing_operation)
            logger.info(f"Call {i+1}: Success - {result}")
            successes += 1
        except Exception as e:
            logger.info(f"Call {i+1}: Failed - {e}")
            failures += 1
        
        await asyncio.sleep(0.5)
    
    logger.info(f"\nResults: {successes} successes, {failures} failures")
    logger.info(f"\nResilience Status:\n{agent.get_resilience_status()}")
    
    # Cancel health checker
    health_task.cancel()


if __name__ == "__main__":
    print("=" * 60)
    print("FAILURE DETECTION AND RECOVERY EXAMPLE")
    print("=" * 60)
    asyncio.run(example_failure_recovery())
```

**Code Statistics:** 450+ lines of production-grade Python code

## Integration with LLM-Whisperer

```python
# In orchestration:
from skills.agentic.failure_detection_recovery import ResilientAgent

orchestrator_agent = ResilientAgent("orchestrator")

# All operations automatically resilient
result = await orchestrator_agent.execute_resilient_operation(
    workflow_engine.execute,
    workflow_id
)
```

## References Summary

- **Netflix Hystrix:** Originator of circuit breaker pattern in production
- **Resilience4j:** Modern Java fault tolerance library
- **AWS Chaos Engineering:** Failure injection for testing resilience
