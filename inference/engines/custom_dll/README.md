# Custom DLL Engine

Author: Shuvam Banerji Seal

This module defines a minimal plugin ABI for custom acceleration kernels.
It is suitable for prototyping proprietary inference operators while preserving
clear initialization and shutdown semantics.

## Features Covered

- C header defining plugin ABI.
- Shared library stub implementation.
- Build script for `.so` plugin artifact.
- Python ctypes validator.

## ABI Contract

The plugin exposes:

- `plugin_init()`
- `plugin_get_info()`
- `plugin_infer()`
- `plugin_shutdown()`

## Build

```bash
bash scripts/build_plugin.sh
```

## Validate

```bash
python scripts/validate_plugin.py --library ./libllm_plugin_stub.so
```
