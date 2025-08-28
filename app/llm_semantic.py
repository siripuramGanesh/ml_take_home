from typing import Dict, List
import openai  # Or any local LLM library
import asyncio

# Configure OpenAI key or environment variable
# openai.api_key = "YOUR_API_KEY"

async def validate_labels_semantic(boxes: Dict[str, List[List[int]]], image_desc: str = "") -> Dict[str, List[List[int]]]:
    """
    Validate or refine labels using LLM.
    boxes: {label: [[x1,y1,x2,y2], ...]}
    image_desc: optional description of image content
    """
    # Flatten labels for LLM prompt
    all_labels = list(boxes.keys())
    prompt = f"""
    I have an image with objects labeled as: {all_labels}.
    Image description (optional): {image_desc}
    Please return a corrected list of labels if any label seems incorrect.
    Format: ["label1", "label2", ...]
    """

    # Call LLM asynchronously
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        text = response.choices[0].message.content
        # Parse list from string (naive)
        corrected_labels = eval(text)
        corrected_boxes = {lbl: boxes[lbl] for lbl in corrected_labels if lbl in boxes}
        return corrected_boxes
    except Exception:
        # fallback to original if LLM fails
        return boxes
