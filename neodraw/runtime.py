import onnxruntime as ort

ort.set_default_logger_severity(3)


class NpuRuntime:
    PROVIDER = "VitisAIExecutionProvider"

    def __init__(self, settings):
        if self.PROVIDER not in ort.get_available_providers():
            raise SystemExit(f"{self.PROVIDER} is not available; run setup.ps1 to install the Ryzen AI wheels")
        if not settings.xclbin.exists():
            raise SystemExit(f"NPU overlay not found: {settings.xclbin}; set sdk or npu_overlay in neodraw.toml")
        self.settings = settings

    def session(self, model, cache_key):
        return ort.InferenceSession(
            str(model),
            providers=[self.PROVIDER],
            provider_options=[{
                "target": self.settings.npu_target,
                "xclbin": str(self.settings.xclbin),
                "cache_dir": str(self.settings.cache),
                "cache_key": cache_key,
                "enable_cache_file_io_in_mem": "0",
            }],
        )
