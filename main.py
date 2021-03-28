
import os, subprocess
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from asgiref.sync import sync_to_async
from fastapi.middleware.cors import CORSMiddleware
from car_numberplate_recognition import initialize_weights, vehicle_detection_video

# Add Environment Variable for instructing the system to run inference on GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Initialize Models Weights
vehicle_net, vehicle_meta, wpod_net, ocr_net, ocr_meta = initialize_weights()

app = FastAPI()
app.mount("/static", StaticFiles(directory="./static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

templates = Jinja2Templates(directory="templates")

# Display Home Webpage
@app.get("/", response_class=HTMLResponse)
async def display_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# Upload and Play Videos
@app.post("/video_upload")
async def video_receive(request: Request):
    body = await request.form()
    video_name = "./static/uploaded_videos/" + body["fileToUpload"].filename
    contents = await body["fileToUpload"].read()

    with open(video_name,"wb") as f:
        f.write(contents)
    
    # Convert Video Format for viewing in a Browser
    command = "ffmpeg -y -i " + video_name + " -c:v libx264 -c:a libfaac -movflags +faststart " + "./static/uploaded_videos/" + ".".join((body["fileToUpload"].filename).split(".")[:-1])+".mp4"
    subprocess.call(command, shell=True)


# Process Uploaded Videos from Clients
@app.post("/process", response_class=HTMLResponse)
async def video_receive(request: Request):
    body = await request.form()
    video_name = "./static/uploaded_videos/"+body["file_name"]
    
    result_video, heads = await sync_to_async(vehicle_detection_video)(video_name, vehicle_net, vehicle_meta, wpod_net, ocr_net, ocr_meta)
    return templates.TemplateResponse("show_result.html", {"request": request, "result_path": result_video, "head_count": heads})

# Download Processed Videos
@app.post("/download", response_class=FileResponse)
async def download_video(file_name: str = Form(...)):
    return FileResponse("./static/results/"+file_name, media_type='application/octet-stream', filename=file_name)

# Enable for providing global access URL
import nest_asyncio
from pyngrok import ngrok
import uvicorn

ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)