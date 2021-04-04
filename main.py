
import os, subprocess
from fastapi                        import FastAPI, Request, Form
from fastapi.responses              import HTMLResponse, FileResponse
from fastapi.staticfiles            import StaticFiles
from fastapi.templating             import Jinja2Templates
from asgiref.sync                   import sync_to_async
from fastapi.middleware.cors        import CORSMiddleware
from car_numberplate_recognition    import initialize_weights, vehicle_detection_video, vehicle_detection_image

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
    # Delete old videos from Server
    try:
        os.remove("./static/uploaded_videos/*.webm")
        os.remove("./static/uploaded_videos/*.jpg")
        os.remove("./static/uploaded_videos/*.png")
        os.remove("./static/uploaded_videos/*.jpeg")
        os.remove("./static/uploaded_videos/*.bmp")
        os.remove("./static/results/*.webm")
        os.remove("./static/results/*.jpg")
        os.remove("./static/results/*.png")
        os.remove("./static/results/*.jpeg")
        os.remove("./static/results/*.bmp")
    except:
        pass

    # Return Home Page from templates
    return templates.TemplateResponse("home.html", {"request": request})


# Upload and Play Videos
@app.post("/video_upload")
async def video_receive(request: Request):
    body = await request.form()
    file_name = "./static/uploaded_videos/" + body["fileToUpload"].filename
    contents = await body["fileToUpload"].read()

    with open(file_name,"wb") as f:
        f.write(contents)


# Process Uploaded Videos/Images from Clients
@app.post("/process", response_class=HTMLResponse)
async def video_receive(request: Request):
    body = await request.form()
    file_name = "./static/uploaded_videos/"+body["file_name"]
    flag = ""

    # Run Inference in Async Mode
    if (file_name.split(".")[-1] in ["jpg", "png", "bmp", "jpeg"]):
        result_file, numberplate_results = await sync_to_async(vehicle_detection_image)(file_name, vehicle_net, vehicle_meta, wpod_net, ocr_net, ocr_meta)
        flag = "img"

    else:
        result_file, numberplate_results = await sync_to_async(vehicle_detection_video)(file_name, vehicle_net, vehicle_meta, wpod_net, ocr_net, ocr_meta)
        flag = "vid"

    return templates.TemplateResponse("show_result.html", {"request": request, "result_path": result_file, "licence_count": numberplate_results, "flag": flag})


# Download Processed Videos/Images
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
