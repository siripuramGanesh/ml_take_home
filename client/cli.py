import asyncio
import json
from pathlib import Path
from typing import Dict, List

import click

from .client import MLClient


@click.group()
def cli():
    """ML TakeHome CLI Client - Production-ready inference and data collection"""
    pass

@cli.command()
@click.option("--image", "-i", required=True, help="Path to image file")
@click.option("--labels", "-l", multiple=True, help="Labels for training")
@click.option("--base64", is_flag=True, help="Use base64 encoding")
@click.option("--output", "-o", help="Output file for results")
def infer(image, labels, base64, output):
    """Run inference on an image"""
    async def run_inference():
        async with MLClient() as client:
            if base64:
                result = await client.infer_base64(image, list(labels))
            else:
                result = await client.infer(image, list(labels))
            
            if output:
                with open(output, 'w') as f:
                    json.dump(result, f, indent=2)
                click.echo(f"Results saved to {output}")
            else:
                click.echo(json.dumps(result, indent=2))
    
    asyncio.run(run_inference())

@cli.command()
@click.option("--image", "-i", required=True, help="Path to image file")
@click.option("--bbox", "-b", "bbox_files", multiple=True, help="Bounding box JSON files")
@click.option("--message", "-m", default="", help="Additional message")
@click.option("--base64", is_flag=True, help="Use base64 encoding")
@click.option("--output", "-o", help="Output file for results")
def collect(image, bbox_files, message, base64, output):
    """Collect data with bounding boxes"""
    bounding_boxes = {}
    for bbox_file in bbox_files:
        try:
            with open(bbox_file, 'r') as f:
                bbox_data = json.load(f)
                if isinstance(bbox_data, dict):
                    bounding_boxes.update(bbox_data)
                else:
                    click.echo(f"Warning: {bbox_file} does not contain a valid dictionary")
        except Exception as e:
            click.echo(f"Error reading {bbox_file}: {e}")
            return
    
    async def run_collect():
        async with MLClient() as client:
            if base64:
                result = await client.collect_base64(image, bounding_boxes, message)
            else:
                result = await client.collect(image, bounding_boxes, message)
            
            if output:
                with open(output, 'w') as f:
                    json.dump(result, f, indent=2)
                click.echo(f"Results saved to {output}")
            else:
                click.echo(json.dumps(result, indent=2))
    
    asyncio.run(run_collect())

@cli.command()
def status():
    """Get model and dataset status"""
    async def run_status():
        async with MLClient() as client:
            model_status = await client.get_model_status()
            dataset_stats = await client.get_dataset_stats()
            
            click.echo("=== Model Status ===")
            click.echo(json.dumps(model_status, indent=2))
            
            click.echo("\n=== Dataset Statistics ===")
            click.echo(json.dumps(dataset_stats, indent=2))
    
    asyncio.run(run_status())

@cli.command()
@click.option("--image", "-i", help="Path to image file for example bbox")
@click.option("--output", "-o", help="Output file for bbox template")
def bbox_template(image, output):
    """Generate bounding box template"""
    template = {
        "person": [[50, 50, 150, 150]],  # [x1, y1, x2, y2]
        "car": [[200, 200, 300, 250]],
        "dog": [[100, 200, 180, 280]]
    }
    
    if image:
        # Add image dimensions to template
        from PIL import Image
        try:
            with Image.open(image) as img:
                width, height = img.size
                template["_image_info"] = {
                    "width": width,
                    "height": height,
                    "filename": Path(image).name
                }
        except Exception as e:
            click.echo(f"Warning: Could not get image info: {e}")
    
    if output:
        with open(output, 'w') as f:
            json.dump(template, f, indent=2)
        click.echo(f"Bounding box template saved to {output}")
    else:
        click.echo(json.dumps(template, indent=2))

@cli.command()
def test():
    """Test connection to servers"""
    async def run_test():
        async with MLClient() as client:
            try:
                model_status = await client.get_model_status()
                dataset_stats = await client.get_dataset_stats()
                
                click.echo("✅ Connected to inference server")
                click.echo("✅ Connected to data collector")
                click.echo(f"📊 Model inferences: {model_status.get('performance_metrics', {}).get('total_inferences', 0)}")
                click.echo(f"📈 Custom images: {dataset_stats.get('custom_images', 0)}")
                
            except Exception as e:
                click.echo(f"❌ Connection failed: {e}")
                click.echo("Make sure servers are running:")
                click.echo("  uvicorn inference.app:app --reload --port 8000")
                click.echo("  uvicorn data_collector.app:app --reload --port 8001")
    
    asyncio.run(run_test())

if __name__ == "__main__":
    cli()