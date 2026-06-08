from dataclasses import dataclass

import yaml


@dataclass
class Config:
    seed: int
    window: int
    version: str


class ConfigLoader:
    @staticmethod
    def load(path: str) -> Config:
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        if "seed" not in data:
            raise ValueError("seed is missing")
        if "window" not in data:
            raise ValueError("window is missing")
        if "version" not in data:
            raise ValueError("version is missing")

        return Config(
            seed=data["seed"],
            window=data["window"],
            version=data["version"],
        )
