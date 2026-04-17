#!/usr/bin/env python3
"""ComfyUI API Client - 連接到 port 8192"""
import requests
import json
import time
import uuid
from pathlib import Path

COMFYUI_URL = "http://127.0.0.1:8192"


def queue_prompt(prompt_dict: dict) -> str:
    """提交 workflow 到佇列，回傳 prompt_id"""
    resp = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": prompt_dict})
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def get_history(prompt_id: str) -> dict:
    """取得 prompt 執行結果"""
    resp = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
    resp.raise_for_status()
    return resp.json().get(prompt_id, {})


def get_queue() -> dict:
    """查看目前佇列狀態"""
    resp = requests.get(f"{COMFYUI_URL}/queue")
    resp.raise_for_status()
    return resp.json()


def wait_for_completion(prompt_id: str, poll_interval: int = 2, timeout: int = 300) -> dict:
    """等待 prompt 執行完成"""
    start = time.time()
    while time.time() - start < timeout:
        history = get_history(prompt_id)
        if history:
            status = history.get("status", {})
            if status.get("completed", False):
                return history
        time.sleep(poll_interval)
    raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout}s")


def get_output_images(history_entry: dict) -> dict[str, list[bytes]]:
    """從 history 取出生成的圖片 binary"""
    images = {}
    outputs = history_entry.get("outputs", {})
    for node_id, node_out in outputs.items():
        if "images" in node_out:
            images[node_id] = []
            for img in node_out["images"]:
                img_resp = requests.get(
                    f"{COMFYUI_URL}/view",
                    params={"filename": img["filename"], "subfolder": img.get("subfolder", "")}
                )
                img_resp.raise_for_status()
                images[node_id].append(img_resp.content)
    return images


def save_images(images: dict[str, list[bytes]], output_dir: str = "outputs"):
    """儲存圖片到硬碟"""
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    saved = []
    for node_id, imgs in images.items():
        for i, img_data in enumerate(imgs):
            fname = f"node_{node_id}_{i:03d}.png"
            (out_path / fname).write_bytes(img_data)
            saved.append(str(out_path / fname))
    return saved


def load_workflow(path: str) -> dict:
    """載入 workflow JSON 檔"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # 快速測試：查詢佇列狀態
    print("=== ComfyUI Status ===")
    print(json.dumps(get_queue(), indent=2))
