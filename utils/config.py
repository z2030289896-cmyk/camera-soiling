"""YAML 配置读取工具。使用 yaml.safe_load 加载配置文件。"""
import yaml


def load_config(path):

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg