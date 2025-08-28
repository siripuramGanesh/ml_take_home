import click
import asyncio
import json
import cv2
import numpy as np
from pathlib import Path
from .api import MLClient  # Import from same package

client = MLClient()

@click.command()
@click.option("--image", required=True, type=click.Path(exists=True), help="Image path")
@click.option("--bbox", multiple=True, type=click.Path(exists=True), help="Bounding box JSON files")
@click.option("--message", default="", help="Optional message for data collector")
@click.option("--infer", is_flag=True, help="Run inference after collect")
def main(image, bbox, message, infer):
    """
    CLI to send image + bounding boxes to /collect and optionally run inference
    """
    asyncio.run(_main(image, bbox, message, infer))

async def _main(image_path, bbox_files, message, do_infer):
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to read image: {image_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Merge all bbox JSON files
    boxes = {}
    for f in bbox_files:
        data = json.loads(Path(f).read_text())
        for k, v in data.items():
            boxes.setdefault(k, []).extend(v)

    if bbox_files:
        print("Sending to collector...")
        res = await client.act(img, boxes, message)
        print("Collector response:", res)

    if do_infer:
        print("Running inference...")
        res = await client.infer(img)
        print("Inference result:", res)

if __name__ == "__main__":
    main()
