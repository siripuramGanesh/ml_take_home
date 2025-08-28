import click
import asyncio
import json
import cv2
import numpy as np
from client.api import MLClient

client = MLClient()

@click.command()
@click.option("--image", required=True, type=click.Path(exists=True), help="Path to the image")
@click.option("--bbox", multiple=True, type=str, help="Bounding box JSON strings or files")
@click.option("--message", default="", help="Optional message for collector")
@click.option("--infer", is_flag=True, help="Run inference after sending to collector")
def main(image, bbox, message, infer):
    asyncio.run(_main(image, bbox, message, infer))

async def _main(image_path, bbox_list, message, do_infer):
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to read image: {image_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Parse bounding boxes
    boxes = {}
    for s in bbox_list:
        try:
            data = json.loads(s)
            if isinstance(data, dict):
                for k, v in data.items():
                    boxes.setdefault(k, []).extend(v)
            else:
                print(f"Skipping invalid bbox (not dict): {s}")
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {s}, error: {e}")

    # Send to collector if boxes exist
    if boxes:
        print("Sending to collector...")
        res = await client.act(img, boxes, message)
        print("Collector response:", res)

    # Optionally run inference
    if do_infer:
        print("Running inference...")
        res = await client.infer(img)
        print("Inference result:", res)

if __name__ == "__main__":
    main()
