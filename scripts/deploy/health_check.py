#!/usr/bin/env python3
"""
Health check script for deployed models.

This script performs health checks on deployed LLM inference
services to verify they're operating correctly. Supports
OpenAI-compatible APIs and custom health endpoints.

Usage:
    python health_check.py --url http://localhost:8000
    python health_check.py --url http://localhost:8000 --timeout 30 --verbose
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class HealthCheckConfig:
    """Configuration for health check."""
    url: str
    timeout: int = 10
    verbose: bool = False
    retries: int = 3
    retry_delay: int = 2


@dataclass
class HealthCheckResult:
    """Result of health check."""
    healthy: bool
    status_code: int
    latency_ms: float
    checks: Dict[str, bool]
    details: Dict[str, Any]


def check_health_endpoint(url: str, timeout: int) -> tuple:
    """Check /health endpoint.

    Args:
        url: Service URL
        timeout: Request timeout

    Returns:
        Tuple of (success, status_code, latency_ms, response)
    """
    health_url = f"{url.rstrip('/')}/health"

    try:
        import requests
    except ImportError:
        logger.error("requests not installed. Install with: pip install requests")
        raise

    start_time = time.time()

    try:
        response = requests.get(health_url, timeout=timeout)
        latency_ms = (time.time() - start_time) * 1000

        return response.status_code == 200, response.status_code, latency_ms, response.text

    except requests.exceptions.Timeout:
        latency_ms = (time.time() - start_time) * 1000
        return False, 0, latency_ms, "Timeout"
    except requests.exceptions.ConnectionError:
        latency_ms = (time.time() - start_time) * 1000
        return False, 0, latency_ms, "Connection error"
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return False, 0, latency_ms, str(e)


def check_models_endpoint(url: str, timeout: int) -> tuple:
    """Check /v1/models endpoint.

    Args:
        url: Service URL
        timeout: Request timeout

    Returns:
        Tuple of (success, status_code, latency_ms, response)
    """
    models_url = f"{url.rstrip('/')}/v1/models"

    try:
        import requests
    except ImportError:
        logger.error("requests not installed")
        raise

    start_time = time.time()

    try:
        response = requests.get(models_url, timeout=timeout)
        latency_ms = (time.time() - start_time) * 1000

        if response.status_code == 200:
            try:
                data = response.json()
                model_count = len(data.get("data", []))
                return True, response.status_code, latency_ms, {"model_count": model_count}
            except json.JSONDecodeError:
                return True, response.status_code, latency_ms, {}

        return False, response.status_code, latency_ms, response.text

    except requests.exceptions.Timeout:
        latency_ms = (time.time() - start_time) * 1000
        return False, 0, latency_ms, "Timeout"
    except requests.exceptions.ConnectionError:
        latency_ms = (time.time() - start_time) * 1000
        return False, 0, latency_ms, "Connection error"
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return False, 0, latency_ms, str(e)


def check_completion_endpoint(url: str, timeout: int) -> tuple:
    """Check completion endpoint with a simple request.

    Args:
        url: Service URL
        timeout: Request timeout

    Returns:
        Tuple of (success, status_code, latency_ms, response)
    """
    completion_url = f"{url.rstrip('/')}/v1/completions"

    try:
        import requests
    except ImportError:
        logger.error("requests not installed")
        raise

    payload = {
        "model": "test",
        "prompt": "Hello",
        "max_tokens": 5,
    }

    start_time = time.time()

    try:
        response = requests.post(
            completion_url,
            json=payload,
            timeout=timeout,
        )
        latency_ms = (time.time() - start_time) * 1000

        return response.status_code in [200, 400, 422], response.status_code, latency_ms, response.text

    except requests.exceptions.Timeout:
        latency_ms = (time.time() - start_time) * 1000
        return False, 0, latency_ms, "Timeout"
    except requests.exceptions.ConnectionError:
        latency_ms = (time.time() - start_time) * 1000
        return False, 0, latency_ms, "Connection error"
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return False, 0, latency_ms, str(e)


def check_metrics_endpoint(url: str, timeout: int) -> tuple:
    """Check /metrics endpoint.

    Args:
        url: Service URL
        timeout: Request timeout

    Returns:
        Tuple of (success, status_code, latency_ms, response)
    """
    metrics_url = f"{url.rstrip('/')}/metrics"

    try:
        import requests
    except ImportError:
        logger.error("requests not installed")
        raise

    start_time = time.time()

    try:
        response = requests.get(metrics_url, timeout=timeout)
        latency_ms = (time.time() - start_time) * 1000

        return response.status_code == 200, response.status_code, latency_ms, response.text

    except requests.exceptions.Timeout:
        latency_ms = (time.time() - start_time) * 1000
        return False, 0, latency_ms, "Timeout"
    except requests.exceptions.ConnectionError:
        latency_ms = (time.time() - start_time) * 1000
        return False, 0, latency_ms, "Connection error"
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        return False, 0, latency_ms, str(e)


def run_health_checks(config: HealthCheckConfig) -> HealthCheckResult:
    """Run all health checks.

    Args:
        config: Health check configuration

    Returns:
        Health check result
    """
    logger.info(f"Running health checks for: {config.url}")

    checks = {}
    overall_healthy = True
    status_code = 0
    total_latency_ms = 0.0
    details = {}

    health_success, status_code, latency, response = check_health_endpoint(
        config.url, config.timeout
    )
    checks["health_endpoint"] = health_success
    total_latency_ms += latency
    if config.verbose:
        details["health"] = {"success": health_success, "status_code": status_code, "latency_ms": latency}

    if not health_success:
        overall_healthy = False

    models_success, status_code, latency, response = check_models_endpoint(
        config.url, config.timeout
    )
    checks["models_endpoint"] = models_success
    total_latency_ms += latency
    if config.verbose:
        details["models"] = {"success": models_success, "status_code": status_code, "latency_ms": latency, "data": response}

    if not models_success:
        overall_healthy = False

    completion_success, status_code, latency, response = check_completion_endpoint(
        config.url, config.timeout
    )
    checks["completion_endpoint"] = completion_success
    total_latency_ms += latency
    if config.verbose:
        details["completion"] = {"success": completion_success, "status_code": status_code, "latency_ms": latency}

    metrics_success, status_code, latency, response = check_metrics_endpoint(
        config.url, config.timeout
    )
    checks["metrics_endpoint"] = metrics_success
    total_latency_ms += latency
    if config.verbose:
        details["metrics"] = {"success": metrics_success, "status_code": status_code, "latency_ms": latency}

    avg_latency = total_latency_ms / max(len(checks), 1)

    return HealthCheckResult(
        healthy=overall_healthy,
        status_code=status_code,
        latency_ms=avg_latency,
        checks=checks,
        details=details,
    )


def main() -> int:
    """Main entry point for health check."""
    parser = argparse.ArgumentParser(
        description="Health check script for deployed models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic health check
    python health_check.py --url http://localhost:8000

    # Health check with verbose output
    python health_check.py --url http://localhost:8000 --verbose

    # Health check with custom timeout
    python health_check.py --url http://localhost:8080 --timeout 30
        """
    )

    parser.add_argument(
        "--url",
        required=True,
        help="Service URL (e.g., http://localhost:8000)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Request timeout in seconds (default: 10)"
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries on failure (default: 3)"
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=2,
        help="Delay between retries in seconds (default: 2)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    config = HealthCheckConfig(
        url=args.url,
        timeout=args.timeout,
        verbose=args.verbose,
        retries=args.retries,
        retry_delay=args.retry_delay,
    )

    result = None

    for attempt in range(args.retries):
        result = run_health_checks(config)

        if result.healthy:
            break

        if attempt < args.retries - 1:
            logger.warning(f"Health check failed, retrying in {args.retry_delay}s... (attempt {attempt + 1}/{args.retries})")
            time.sleep(args.retry_delay)

    if args.verbose:
        print(json.dumps({
            "healthy": result.healthy,
            "latency_ms": result.latency_ms,
            "checks": result.checks,
            "details": result.details,
        }, indent=2))
    else:
        status = "HEALTHY" if result.healthy else "UNHEALTHY"
        print(f"Status: {status}")
        print(f"Latency: {result.latency_ms:.2f}ms")
        print(f"Checks: {json.dumps(result.checks)}")

    return 0 if result.healthy else 1


if __name__ == "__main__":
    sys.exit(main())