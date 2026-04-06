# Performance Profiling & Optimization: Identifying and Eliminating Bottlenecks

**Author**: Shuvam Banerji Seal  
**Category**: Foundational Skills  
**Difficulty**: Intermediate to Advanced  
**Last Updated**: April 2026

## Problem Statement

Python applications can suffer from performance issues in production. This skill provides:
- **CPU Profiling**: Identify hot functions consuming CPU cycles
- **Memory Profiling**: Detect memory leaks and inefficient allocations
- **Bottleneck Analysis**: Systematically find performance issues
- **Benchmarking**: Quantify improvements
- **Optimization Strategies**: Apply proven techniques
- **Production Profiling**: Monitor real-world performance

---

## Theoretical Foundations

### 1. Amdahl's Law

```
Speedup = 1 / ((1 - P) + P/N)

Where:
  P = Fraction of program parallelizable [0-1]
  N = Number of processors
  
Example:
  If 80% of code is parallelizable (P=0.8) and N=4:
  Speedup = 1 / ((1-0.8) + 0.8/4) = 1 / (0.2 + 0.2) = 2.5x
```

**Implication**: Optimizing non-parallelizable code yields diminishing returns.

### 2. Performance Profiling Methodology

```
Observe → Measure → Analyze → Optimize → Repeat

1. Observe: Run application, note slow operations
2. Measure: Profile to get data
3. Analyze: Identify bottlenecks
4. Optimize: Change implementation
5. Repeat: Verify improvement, look for next bottleneck
```

### 3. Memory Profiling Model

```
Total Memory = Fixed Overhead + Data Structures + Cache
Memory Leak = Unreleased References = Garbage Accumulation

Detection:
  Baseline = Memory at start
  Current = Memory after operation
  Leak Detected if: (Current - Baseline) > Expected
```

---

## Comprehensive Code Examples

### Example 1: CPU Profiling with cProfile

```python
import cProfile
import pstats
import io
from typing import List
import time

def inefficient_search(items: List[int], target: int) -> int:
    """Inefficient: Linear search with redundant operations."""
    for i in range(len(items)):
        # Redundant computation
        _ = sum(items[:i])  # Compute sum each iteration
        if items[i] == target:
            return i
    return -1


def efficient_search(items: List[int], target: int) -> int:
    """Efficient: Direct comparison."""
    for i, item in enumerate(items):
        if item == target:
            return i
    return -1


def fibonacci_recursive(n: int) -> int:
    """Inefficient: Exponential time complexity O(2^n)."""
    if n < 2:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci_memoized(n: int, memo: dict[int, int] | None = None) -> int:
    """Efficient: Linear time complexity O(n) with memoization."""
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n < 2:
        return n
    memo[n] = fibonacci_memoized(n - 1, memo) + fibonacci_memoized(n - 2, memo)
    return memo[n]


class Profiler:
    """Context manager for CPU profiling."""
    
    def __init__(self, name: str = "Profile"):
        self.name = name
        self.pr = cProfile.Profile()
    
    def __enter__(self):
        self.pr.enable()
        return self
    
    def __exit__(self, *args):
        self.pr.disable()
        self.print_stats()
    
    def print_stats(self):
        """Print profiling statistics."""
        s = io.StringIO()
        ps = pstats.Stats(self.pr, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions
        
        print(f"\n{'='*60}")
        print(f"{self.name}")
        print(f"{'='*60}")
        print(s.getvalue())


# Profiling examples
if __name__ == "__main__":
    # Example 1: Search comparison
    items = list(range(10000))
    target = 9999
    
    print("\n--- CPU Profiling Comparison ---\n")
    
    with Profiler("Inefficient Search"):
        result = inefficient_search(items, target)
    
    with Profiler("Efficient Search"):
        result = efficient_search(items, target)
    
    # Example 2: Fibonacci comparison
    print("\n--- Fibonacci Comparison ---\n")
    
    with Profiler("Recursive Fibonacci(25)"):
        result = fibonacci_recursive(25)
    
    with Profiler("Memoized Fibonacci(25)"):
        result = fibonacci_memoized(25)
    
    # Example 3: Line-by-line profiling
    print("\n--- Line-by-Line Profiling ---")
    print("Install: pip install line_profiler")
    print("Usage: kernprof -l -v script.py")
```

### Example 2: Memory Profiling

```python
import tracemalloc
from typing import List
import sys
from memory_profiler import profile

class MemoryProfiler:
    """Context manager for memory profiling."""
    
    def __init__(self, name: str = "Memory Profile"):
        self.name = name
        self.start_memory = 0
    
    def __enter__(self):
        tracemalloc.start()
        self.start_memory = tracemalloc.get_traced_memory()[0]
        return self
    
    def __exit__(self, *args):
        current, peak = tracemalloc.get_traced_memory()
        used = (current - self.start_memory) / 1024 / 1024  # MB
        peak_mb = peak / 1024 / 1024
        
        print(f"\n{self.name}")
        print(f"  Memory used: {used:.2f} MB")
        print(f"  Peak memory: {peak_mb:.2f} MB")
        tracemalloc.stop()


def memory_inefficient() -> List[int]:
    """Inefficient: Creates many intermediate lists."""
    result = []
    for i in range(1000000):
        # Creates new list each iteration
        temp = [i] * 100
        result.extend(temp)
    return result


def memory_efficient() -> List[int]:
    """Efficient: Preallocates space."""
    result = [0] * 100000000  # Preallocate
    for i in range(1000000):
        result[i] = i
    return result[:1000000]


@profile  # Requires line_profiler
def function_with_memory_profile(n: int) -> int:
    """Function for line-by-line memory profiling."""
    total = 0
    items = []
    
    for i in range(n):
        items.append(i)  # Memory allocation
        total += i
    
    return total


# Detecting memory leaks
class LeakyCache:
    """Example of memory leak: unbounded cache."""
    
    def __init__(self):
        self._cache: dict = {}
    
    def get(self, key: str) -> str:
        if key not in self._cache:
            self._cache[key] = f"value_{key}"  # Never cleared!
        return self._cache[key]


class FixedCache:
    """Fixed: Bounded cache with LRU eviction."""
    from functools import lru_cache
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache = {}
    
    def get(self, key: str) -> str:
        if len(self._cache) >= self.max_size:
            # Remove oldest (simplistic)
            self._cache.pop(next(iter(self._cache)))
        
        if key not in self._cache:
            self._cache[key] = f"value_{key}"
        return self._cache[key]


# Profiling examples
if __name__ == "__main__":
    print("\n--- Memory Efficiency Comparison ---\n")
    
    with MemoryProfiler("Inefficient Memory"):
        data = memory_inefficient()
    
    with MemoryProfiler("Efficient Memory"):
        data = memory_efficient()
    
    # Leak detection
    print("\n--- Memory Leak Detection ---\n")
    
    with MemoryProfiler("Leaky Cache"):
        cache = LeakyCache()
        for i in range(100000):
            cache.get(f"key_{i}")
    
    with MemoryProfiler("Fixed Cache"):
        cache = FixedCache(max_size=1000)
        for i in range(100000):
            cache.get(f"key_{i}")
```

### Example 3: Benchmarking with pytest-benchmark

```python
import pytest
import time
from typing import List
import numpy as np

def quicksort(arr: List[int]) -> List[int]:
    """Quick sort implementation."""
    if len(arr) <= 1:
        return arr
    
    pivot = arr[0]
    less = [x for x in arr[1:] if x <= pivot]
    greater = [x for x in arr[1:] if x > pivot]
    
    return quicksort(less) + [pivot] + quicksort(greater)


def builtin_sort(arr: List[int]) -> List[int]:
    """Built-in sort (highly optimized)."""
    return sorted(arr)


class TestBenchmarks:
    """Benchmark tests using pytest-benchmark."""
    
    def test_quicksort_small(self, benchmark):
        """Benchmark quicksort on small dataset."""
        data = list(range(100))
        result = benchmark(quicksort, data)
        assert len(result) == 100
    
    def test_builtin_sort_small(self, benchmark):
        """Benchmark built-in sort on small dataset."""
        data = list(range(100))
        result = benchmark(builtin_sort, data)
        assert len(result) == 100
    
    def test_quicksort_large(self, benchmark):
        """Benchmark quicksort on large dataset."""
        data = list(range(10000))
        result = benchmark(quicksort, data)
        assert len(result) == 10000
    
    def test_builtin_sort_large(self, benchmark):
        """Benchmark built-in sort on large dataset."""
        data = list(range(10000))
        result = benchmark(builtin_sort, data)
        assert len(result) == 10000
    
    @pytest.mark.parametrize("size", [100, 1000, 10000])
    def test_sort_scaling(self, benchmark, size):
        """Benchmark sort with different sizes."""
        data = list(range(size))
        result = benchmark(builtin_sort, data)
        assert len(result) == size


# Manual benchmarking
def manual_benchmark(func, *args, iterations=1000, **kwargs):
    """Simple manual benchmarking utility."""
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    
    times.sort()
    
    return {
        "min": min(times),
        "max": max(times),
        "avg": sum(times) / len(times),
        "median": times[len(times) // 2]
    }


# Optimization example
class DataProcessor:
    """Example with optimization opportunities."""
    
    @staticmethod
    def process_unoptimized(data: List[int]) -> int:
        """Unoptimized: Multiple passes through data."""
        total = 0
        count = 0
        max_val = float('-inf')
        
        for item in data:
            total += item
        
        for item in data:
            count += 1
        
        for item in data:
            if item > max_val:
                max_val = item
        
        return total + count + max_val
    
    @staticmethod
    def process_optimized(data: List[int]) -> int:
        """Optimized: Single pass through data."""
        total = 0
        count = 0
        max_val = float('-inf')
        
        for item in data:
            total += item
            count += 1
            if item > max_val:
                max_val = item
        
        return total + count + max_val


# Run benchmarks
if __name__ == "__main__":
    print("\n--- Manual Benchmarking ---\n")
    
    data = list(range(1000))
    
    unopt_time = manual_benchmark(
        DataProcessor.process_unoptimized,
        data,
        iterations=1000
    )
    
    opt_time = manual_benchmark(
        DataProcessor.process_optimized,
        data,
        iterations=1000
    )
    
    print(f"Unoptimized: {unopt_time['avg']*1e6:.2f} µs")
    print(f"Optimized:   {opt_time['avg']*1e6:.2f} µs")
    print(f"Speedup:     {unopt_time['avg']/opt_time['avg']:.2f}x")
```

### Example 4: Flame Graph Generation

```python
import py_spy
import time
from typing import Callable
import subprocess

def generate_flame_graph(
    script_path: str,
    output_path: str = "flame_graph.svg"
):
    """
    Generate flame graph for a Python script.
    
    Requires: pip install py-spy flamegraph
    
    Flame graph visualization:
    - X-axis: Function call frequency
    - Y-axis: Call stack depth
    - Width: Time spent
    """
    # Record profile
    subprocess.run([
        "py-spy", "record",
        "-o", f"{output_path}.prof",
        "python", script_path
    ])
    
    # Convert to flame graph (requires FlameGraph tools)
    # stackcollapse-perf {output_path}.prof > {output_path}.folded
    # flamegraph.pl {output_path}.folded > {output_path}.svg


def profile_with_sampling(
    func: Callable,
    duration: float = 10
):
    """
    Profile function with statistical sampling.
    
    py-spy uses sampling (non-intrusive):
    - Periodically checks call stack
    - Minimal overhead
    - Suitable for production profiling
    """
    print(f"Profiling {func.__name__} for {duration}s...")
    time.sleep(0.1)  # Give process time to start
    print("Complete!")
```

### Example 5: Comprehensive Optimization Workflow

```python
from dataclasses import dataclass
from typing import List, Dict
import time

@dataclass
class Profile:
    """Profiling results."""
    name: str
    time_ms: float
    memory_mb: float


class PerformanceAnalysis:
    """Framework for performance analysis and optimization."""
    
    def __init__(self):
        self.profiles: List[Profile] = []
    
    def measure(self, name: str, func: Callable, *args, **kwargs) -> any:
        """Measure function performance."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start_time) * 1000  # ms
        
        profile = Profile(name=name, time_ms=elapsed, memory_mb=0)
        self.profiles.append(profile)
        
        return result
    
    def compare_implementations(self, 
                              implementations: Dict[str, Callable],
                              *args, **kwargs):
        """Compare multiple implementations."""
        results = {}
        
        for name, func in implementations.items():
            _ = self.measure(name, func, *args, **kwargs)
        
        self._print_comparison()
        return results
    
    def _print_comparison(self):
        """Print comparison table."""
        if not self.profiles:
            return
        
        fastest = min(self.profiles, key=lambda p: p.time_ms)
        
        print(f"\n{'Name':<30} {'Time (ms)':<15} {'Speedup':<10}")
        print("-" * 55)
        
        for profile in self.profiles:
            speedup = fastest.time_ms / profile.time_ms
            print(f"{profile.name:<30} {profile.time_ms:<15.4f} {speedup:<10.2f}x")


# Usage example
def example_workflow():
    """Complete optimization workflow."""
    
    # Original implementation
    def process_data_v1(data: List[int]) -> int:
        result = 0
        for item in data:
            result += item * 2  # Unnecessary computation
        return result
    
    # Optimized v2: Reduce computation
    def process_data_v2(data: List[int]) -> int:
        return sum(item * 2 for item in data)
    
    # Optimized v3: Use built-in
    def process_data_v3(data: List[int]) -> int:
        return 2 * sum(data)
    
    # Measure
    data = list(range(100000))
    analyzer = PerformanceAnalysis()
    
    analyzer.compare_implementations({
        "v1_loop": process_data_v1,
        "v2_comprehension": process_data_v2,
        "v3_builtin": process_data_v3,
    }, data)


if __name__ == "__main__":
    example_workflow()
```

---

## Step-by-Step Implementation Guide

### 1. CPU Profiling Workflow

**Step 1.1: Run cProfile**
```python
python -m cProfile -s cumulative script.py
```

**Step 1.2: Analyze results**
```
ncalls tottime percall cumtime percall filename:lineno
     5    0.001   0.000    0.001   0.000 script.py:10
```

**Step 1.3: Optimize hottest functions**
- Focus on cumulative time (cumtime)
- Optimize functions called most frequently
- Consider algorithm improvements

### 2. Memory Profiling Workflow

**Step 2.1: Install tools**
```bash
pip install memory-profiler line_profiler
```

**Step 2.2: Decorate functions**
```python
@profile
def memory_intensive():
    pass
```

**Step 2.3: Run profiler**
```bash
python -m memory_profiler script.py
```

### 3. Benchmarking

**Step 3.1: Install pytest-benchmark**
```bash
pip install pytest-benchmark
```

**Step 3.2: Write benchmark tests**
```python
def test_function(benchmark):
    result = benchmark(function, arg1, arg2)
```

**Step 3.3: Run and compare**
```bash
pytest --benchmark-only
pytest --benchmark-compare
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Premature Optimization
**Problem**: Optimize before profiling
```python
# Don't optimize without data!
result = [x*2 for x in data]  # Is this slow?
```

**Solution**: Profile first
```python
with Profiler():
    result = [x*2 for x in data]
# If it's not a bottleneck, don't optimize
```

### Pitfall 2: Ignoring Profiler Overhead
**Problem**: Profiler overhead skews results
```python
# cProfile has overhead
cProfile.run('expensive_function()')
```

**Solution**: Use sampling profilers for production
```python
# py-spy has minimal overhead
py-spy record -o output.prof python script.py
```

### Pitfall 3: Optimization Breaking Semantics
**Problem**: Optimization changes behavior
```python
# Original
def safe_divide(a, b):
    if b == 0:
        return 0
    return a / b

# "Optimized" (wrong!)
def optimized(a, b):
    return a / b if b else 0  # Different logic?
```

**Solution**: Maintain test coverage during optimization

---

## Performance Benchmarks

```
Operation              Time (microseconds)
Function call         0.05 µs
List comprehension    2 µs per element
Loop iteration        0.5 µs
Dict lookup           0.1 µs
```

---

## Integration with LLM Systems

### 1. LLM Inference Profiling
```python
with Profiler("LLM Inference"):
    output = model.generate(prompt)
```

### 2. Batch Processing Optimization
```python
# Profile batch vs. sequential
analyzer.compare_implementations({
    "sequential": process_sequential,
    "batch": process_batch,
}, data)
```

---

## Authoritative Sources

1. **cProfile documentation**: https://docs.python.org/3/library/profile.html
2. **memory_profiler**: https://github.com/pythonprofilers/memory_profiler
3. **line_profiler**: https://github.com/pyutils/line_profiler
4. **py-spy**: https://github.com/benfred/py-spy
5. **pytest-benchmark**: https://pytest-benchmark.readthedocs.io/
6. **Flamegraph**: http://www.brendangregg.com/flamegraphs.html
7. **Python Performance Tips**: https://wiki.python.org/moin/PythonSpeed
8. **Algorithmic Complexity**: https://en.wikipedia.org/wiki/Time_complexity
9. **Systems Performance by Brendan Gregg**: https://www.oreilly.com/library/view/systems-performance-2nd/9780136820239/
10. **Amdahl's Law**: https://en.wikipedia.org/wiki/Amdahl%27s_law

---

## Summary

Optimize systematically through:
- CPU profiling to identify hot functions
- Memory profiling to detect leaks
- Benchmarking to quantify improvements
- Data-driven optimization
- Production profiling for real-world issues

These patterns enable high-performance Python systems for LLM inference, batch processing, and production services.
