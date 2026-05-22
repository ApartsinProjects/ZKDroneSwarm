"""Generate hero image for ZK-MRTA README."""
import json
import sys
from pathlib import Path
from google import genai
from google.genai import types

config = json.loads(Path.home().joinpath(".gemini-imagegen.json").read_text())
client = genai.Client(api_key=config["api_key"])

PROMPT = (
    "Cinematic ultra-wide hero image for a computer science research paper on autonomous drone swarm coordination. "
    "Top-down aerial night view of a dark tactical terrain. "
    "A swarm of 9 small autonomous drones (sleek hexagonal glowing blue-white bodies, softly lit) "
    "are dispersed across the scene in a loose formation. "
    "Each drone has a single faint dotted line connecting it to its own chosen target — "
    "there are NO lines between drones, emphasizing zero communication. "
    "Targets are 27 small red-orange hexagonal nodes scattered in clusters on the dark ground. "
    "Subtle data overlays: faint matrix symbols, latent vector notation (z_i, z_j), cosine similarity formula. "
    "Color palette: deep navy black background, electric blue drones, ember-orange targets, cyan data overlays. "
    "Cinematic depth-of-field, photorealistic, ultra-wide 16:9."
)

print("Generating hero image with Imagen 4...")
response = client.models.generate_images(
    model="imagen-4.0-generate-001",
    prompt=PROMPT,
    config=types.GenerateImagesConfig(
        number_of_images=1,
        aspect_ratio="16:9",
        output_mime_type="image/png",
    ),
)

out_path = Path("E:/Projects/ColabDroneSwarm/docs/hero.png")
out_path.parent.mkdir(parents=True, exist_ok=True)

for img in response.generated_images:
    image_bytes = img.image.image_bytes
    out_path.write_bytes(image_bytes)
    print(f"Saved: {out_path} ({len(image_bytes):,} bytes)")

print("Done.")
