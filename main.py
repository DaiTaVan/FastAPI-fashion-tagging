import uvicorn
import cv2
import numpy as np
from utils import FashionTagging, ErrorCode
import time
from fastapi import  UploadFile,  FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

modelEngine = FashionTagging(model_path= 'mask_rcnn_model')

@app.post("/upload_image/")
async def upload(input_image: UploadFile = File(...)):

    if input_image.file is None:
        return ErrorCode("Vui lòng truyền dữ liệu ảnh vào formdata")
    
    # check the content type (MIME type)
    content_type = input_image.content_type
    if content_type not in ["image/jpeg", "image/png", "image/gif"]:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    imgStr = input_image.file.read()
    
    
    # npimg = np.fromstring(imgStr, np.uint8)
    npimg = np.frombuffer(imgStr, np.uint8)
    
    image = cv2.imdecode(npimg, flags=1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    prediction = modelEngine.process_one_image(image)
    
    return {"message": prediction}

if __name__=="__main__":
    uvicorn.run(app,host="127.0.0.1",port=8000)