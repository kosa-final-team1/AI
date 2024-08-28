# from fastapi import FastAPI

# app = FastAPI()
from fastapi import FastAPI
import subprocess
import time
#from fastapi import FastAPI, Request
import json

app = FastAPI()

@app.get("/chat")
async def chat():
    return {"message":"hello FastAPI"} 

# Streamlit을 서브 프로세스로 실행하는 함수
def run_streamlit():
    #subprocess.Popen(["streamlit", "run", "mm.py"])
    # http://0.0.0.0:8501/
     subprocess.Popen(["streamlit", "run", "fashionbot_0826.py", "--server.address", "0.0.0.0", "--server.port", "8501"])
    # subprocess.Popen(["streamlit", "run", "fashionbot_0826.py", "--server.address", "127.0.0.1", "--server.port", "8501"])
async def startup_event():
    # FastAPI 서버가 시작될 때 Streamlit을 실행
    run_streamlit()
    time.sleep(5)  # Streamlit이 구동되기까지 잠시 대기

@app.get("/status")
def read_root():
    return {"message": "****** FastAPI 구동중 확인 완료 ******"}
app = FastAPI()

# @app.post("/chat")
# async def receive_user_data(request: Request):
#     try:
#         # JSON 데이터를 읽어옵니다.
#         data = await request.json()
        
#         # 받은 데이터를 처리합니다.
#         user_id = data.get("userId")
#         #additional_data = data.get("additionalData")
        
#         # 데이터를 사용하여 원하는 작업 수행
#         #print(f"Received userId: {user_id}, additionalData: {additional_data}")
#         print(f"Received userId: {user_id}")

#         # 응답을 반환
#         return {"status": "success", "userId": user_id}
#     except Exception as e:
#         print(f"Error: {e}")
#         return {"status": "error", "message": str(e)}
    
    
#127.0.0.1:8000/streamlit-status
@app.get("/streamlit-status")
def check_streamlit_status():
    # Streamlit이 실행 중인지 확인하는 엔드포인트
    try:
        response = subprocess.check_output(["pgrep", "-f", "streamlit"])
        if response:
            return {"status": "****** Streamlit 구동중 ******"}
    except subprocess.CalledProcessError:
        return {"status": "****** 구동 안 됨 ******"}

    return {"status": "Unknown"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)