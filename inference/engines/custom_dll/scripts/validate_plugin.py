import argparse
import ctypes
import json
import sys


class PluginInfo(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("version", ctypes.c_int)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate custom DLL plugin ABI")
    parser.add_argument("--library", required=True)
    args = parser.parse_args()

    lib = ctypes.CDLL(args.library)

    lib.plugin_init.restype = ctypes.c_int
    lib.plugin_get_info.restype = ctypes.POINTER(PluginInfo)
    lib.plugin_infer.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    lib.plugin_infer.restype = ctypes.c_int
    lib.plugin_shutdown.restype = None

    if lib.plugin_init() != 0:
        raise RuntimeError("plugin_init returned non-zero")

    info = lib.plugin_get_info().contents

    in_values = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
    out_values = (ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0)
    count = lib.plugin_infer(in_values, 4, out_values, 4)

    lib.plugin_shutdown()

    report = {
        "plugin_name": info.name.decode("utf-8"),
        "plugin_version": info.version,
        "processed": count,
        "output": [float(v) for v in out_values],
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
