from fastapi import FastAPI, UploadFile, Form
from google.cloud import storage
from google import genai
from io import BytesIO
from PIL import Image
import uuid

app = FastAPI()

# Initialize GCP clients (Credentials are auto-detected on Cloud Run)
storage_client = storage.Client()
bucket = storage_client.bucket("tryon-assets-bucket") # Replace with your bucket
ai_client = genai.Client() 

@app.post("/api/generate")
async def generate_tryon(
    file: UploadFile, 
    ethnicity: str = Form(...), 
    background: str = Form(...)
):
    image_bytes = await file.read()
    
    # 1. Prepare the exact prompt
    prompt = (
        f"Create a photorealistic, full-body portrait of a {ethnicity} fashion model "
        f"wearing the exact clothing item shown in the reference image. "
        f"The background setting must be {background}. "
        "Strictly preserve the texture, drape, color, and cut of the original garment."
    )
    
    # 2. Call Nano Banana Pro via the Google GenAI SDK
    ref_image = Image.open(BytesIO(image_bytes))
    response = ai_client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=[prompt, ref_image]
    )
    
    # 3. Save the generated image directly to Cloud Storage
    output_filename = f"outputs/model_{uuid.uuid4()}.png"
    output_blob = bucket.blob(output_filename)
    
    for part in response.parts:
        if part.inline_data:
            output_blob.upload_from_string(
                part.inline_data.data, 
                content_type="image/png"
            )
            return {"status": "success", "image_url": output_blob.public_url}
            
    return {"status": "error", "message": "Model failed to generate an image."}