import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from deep_translator import GoogleTranslator
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from PIL import Image as PILImage
from io import BytesIO
import numpy as np
import os
import torch
from dotenv import load_dotenv
import chromadb
from typing import TypedDict

# .env 파일 로드
load_dotenv()

# 환경 변수에서 API 키 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("[Error] OpenAI API Key를 찾을 수 없습니다..")

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fashion-CLIP 모델 설정
processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
model = AutoModelForZeroShotImageClassification.from_pretrained("patrickjohncyh/fashion-clip").to(device)

# ChromaDB 클라이언트 초기화
local_path = '/Users/jei/Downloads/'  
chroma_db = 'chromadb_j_0816'
client = chromadb.PersistentClient(path=local_path + chroma_db)

# 컬렉션 로드
image_collection = client.get_collection("outfit_img_embeddings")
text_collection = client.get_collection("outfit_txt_embeddings")
products_collection = client.get_collection("products_metadatas")

# 세션 상태 정의
class GraphState(TypedDict):
    session_id: str
    user_input_text: str
    translated_text: str
    user_input_image: bytes
    user_intent: str
    input_type: str
    retrieved_metadata: dict
    chatbot_answer: str

# 세션 상태 초기화
store = {}

def save_to_session_history(session_id, interaction):
    if session_id not in store:
        store[session_id] = []
    store[session_id].append(interaction)

# 텍스트 번역 함수
def trans_user_input(state: GraphState):
    user_input = state["user_input_text"]
    translator = GoogleTranslator(source="auto", target="en")
    translated_text = translator.translate(user_input)
    state["translated_text"] = translated_text
    return state

# 멀티모달 추천 함수
def multi_recommend(state: GraphState):
    user_image_path = state["user_input_image"]
    text_input = state["translated_text"]

    # 이미지 열기
    image = PILImage.open(BytesIO(user_image_path)).convert("RGB")

    # 이미지 임베딩 생성
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs).cpu().numpy()

    # 임베딩 정규화
    image_embedding = image_embedding / np.linalg.norm(image_embedding, ord=2, axis=-1, keepdims=True)

    # Chroma DB에서 상위 30개의 유사한 데이터를 검색
    results = image_collection.query(
        query_embeddings=image_embedding.tolist(),
        n_results=30,
        include=["embeddings", "metadatas", "distances"]
    )

    if not results['metadatas']:
        return state

    # 텍스트 임베딩 생성
    text_inputs = processor(text=[text_input], return_tensors="pt").to(device)
    with torch.no_grad():
        text_embedding = model.get_text_features(**text_inputs).cpu().numpy()

    # 텍스트 임베딩 정규화
    text_embedding = text_embedding / np.linalg.norm(text_embedding, ord=2, axis=-1, keepdims=True)

    # 텍스트와 상위 30개 이미지의 유사도 계산
    similarities = np.dot(np.array(results['embeddings'][0]), text_embedding.T)

    best_match_idx = np.argmax(similarities)
    if best_match_idx >= len(results['metadatas'][0]):
        return state

    # 결과 저장
    state["retrieved_metadata"] = results['metadatas'][0][best_match_idx]
    state["retrieved_distance"] = similarities[best_match_idx]

    return state

# 챗봇 응답 생성 함수
def chatbot_output(state: GraphState):
    output_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    metadata = state["retrieved_metadata"]
    gpt_input = f"이 패션 아이템은 {metadata}. 이 데이터를 바탕으로 사용자에게 추천하는 설명을 생성해 주세요."
    explanation = output_llm.invoke(gpt_input).content
    state["chatbot_answer"] = explanation
    return state

# Streamlit UI 구성
st.title("AI Fashion Chatbot")

session_id = st.text_input("세션 ID를 입력하세요:", "default_session")
user_input_text = st.text_input("텍스트로 질문하세요:")
user_input_image = st.file_uploader("이미지를 업로드하세요:")

if st.button("전송"):
    state = GraphState(
        session_id=session_id,
        user_input_text=user_input_text,
        translated_text=None,
        user_input_image=None,
        user_intent=None,
        input_type=None,
        retrieved_metadata={},
        chatbot_answer=None
    )

    if user_input_image:
        state["user_input_image"] = user_input_image.read()

    if user_input_text:
        state = trans_user_input(state)

    state = multi_recommend(state)

    state = chatbot_output(state)

    # Chatbot's response displayed in chat format
    st.write(f"Chatbot's response: {state['chatbot_answer']}")

    if state["retrieved_metadata"]:
        image_url = state["retrieved_metadata"].get("outfit_img_url")
        if image_url:
            st.image(image_url, caption="추천 이미지")
        else:
            st.write("이미지를 찾을 수 없습니다.")