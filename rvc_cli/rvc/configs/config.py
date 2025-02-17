import torch
import json
import os

CONFIG_BASE_PATH = os.path.join("rvc_cli", "rvc", "configs")

version_config_paths = [
    os.path.join("v1", "32000.json"),
    os.path.join("v1", "40000.json"),
    os.path.join("v1", "48000.json"),
    os.path.join("v2", "48000.json"),
    os.path.join("v2", "40000.json"),
    os.path.join("v2", "32000.json"),
]


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class Config:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.is_half = self.device.startswith("cuda")
        self.gpu_name = (
            torch.cuda.get_device_name(int(self.device.split(":")[-1]))
            if self.device.startswith("cuda")
            else None
        )
        self.json_config = self.load_config_json()
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def load_config_json(self) -> dict:
        configs = {}
        for config_file in version_config_paths:
            config_path = os.path.join(CONFIG_BASE_PATH, config_file)

            if not os.path.exists(config_path):
                print(f"[WARNING] Config file not found: {config_path}")
                continue  # Skip missing config files

            try:
                with open(config_path, "r") as f:
                    configs[config_file] = json.load(f)
            except json.JSONDecodeError:
                print(f"[ERROR] Failed to parse JSON in {config_path}")

        return configs

    def has_mps(self) -> bool:
        return torch.backends.mps.is_available()

    def has_xpu(self) -> bool:
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    def set_precision(self, precision):
        if precision not in ["fp32", "fp16"]:
            raise ValueError("Invalid precision type. Must be 'fp32' or 'fp16'.")

        fp16_run_value = precision == "fp16"
        for config_path in version_config_paths:
            full_config_path = os.path.join(CONFIG_BASE_PATH, config_path)
            if not os.path.exists(full_config_path):
                print(f"[WARNING] Config file missing: {full_config_path}")
                continue

            try:
                with open(full_config_path, "r") as f:
                    config = json.load(f)
                config["train"]["fp16_run"] = fp16_run_value
                with open(full_config_path, "w") as f:
                    json.dump(config, f, indent=4)
            except (FileNotFoundError, json.JSONDecodeError):
                print(f"[ERROR] Failed to update {full_config_path}")

        return f"Set precision to {precision} in available config files."

    def get_precision(self):
        if not version_config_paths:
            raise FileNotFoundError("No configuration paths provided.")

        full_config_path = os.path.join(CONFIG_BASE_PATH, version_config_paths[0])
        if not os.path.exists(full_config_path):
            print(f"[ERROR] Config file missing: {full_config_path}")
            return None

        try:
            with open(full_config_path, "r") as f:
                config = json.load(f)
            return "fp16" if config["train"].get("fp16_run", False) else "fp32"
        except json.JSONDecodeError:
            print(f"[ERROR] JSON parsing failed in {full_config_path}")
            return None

    def device_config(self) -> tuple:
        if self.device.startswith("cuda"):
            self.set_cuda_config()
        elif self.has_mps():
            self.device = "mps"
            self.is_half = False
            self.set_precision("fp32")
        else:
            self.device = "cpu"
            self.is_half = False
            self.set_precision("fp32")

        x_pad, x_query, x_center, x_max = (
            (3, 10, 60, 65) if self.is_half else (1, 6, 38, 41)
        )
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad, x_query, x_center, x_max = (1, 5, 30, 32)

        return x_pad, x_query, x_center, x_max

    def set_cuda_config(self):
        i_device = int(self.device.split(":")[-1])
        self.gpu_name = torch.cuda.get_device_name(i_device)
        low_end_gpus = ["16", "P40", "P10", "1060", "1070", "1080"]
        if (
            any(gpu in self.gpu_name for gpu in low_end_gpus)
            and "V100" not in self.gpu_name.upper()
        ):
            self.is_half = False
            self.set_precision("fp32")

        self.gpu_mem = torch.cuda.get_device_properties(i_device).total_memory // (
            1024**3
        )


def max_vram_gpu(gpu):
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(gpu)
        return round(gpu_properties.total_memory / 1024 / 1024 / 1024)
    return 8


def get_gpu_info():
    ngpu = torch.cuda.device_count()
    gpu_infos = []
    if torch.cuda.is_available() or ngpu != 0:
        for i in range(ngpu):
            gpu_name = torch.cuda.get_device_name(i)
            mem = int(torch.cuda.get_device_properties(i).total_memory / 1024**3 + 0.4)
            gpu_infos.append(f"{i}: {gpu_name} ({mem} GB)")
    return "\n".join(gpu_infos) if gpu_infos else "No compatible GPU found."


def get_number_of_gpus():
    return "-".join(map(str, range(torch.cuda.device_count()))) if torch.cuda.is_available() else "-"
