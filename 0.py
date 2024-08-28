import streamlit as st
from typing import TypedDict
from deep_translator import GoogleTranslator
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_teddynote import logging
import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from dotenv import load_dotenv
import numpy as np
import os
import chromadb
from PIL import Image as PILImage
from io import BytesIO

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fashion-CLIP 모델 설정
processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip")
model = AutoModelForZeroShotImageClassification.from_pretrained(
    "patrickjohncyh/fashion-clip"
).to(device)

# .env 파일 로드
load_dotenv()

# 환경 변수에서 API 키 가져오기
# openai_api_key = os.getenv("OPENAI_API_KEY")

# 프로젝트 이름을 입력합니다.
logging.langsmith("test_01")

# ChromaDB 클라이언트 초기화
local_path = "/Users/choibyounghoon/Desktop/test/"
chroma_db = "chromadb_j_0816"
client = chromadb.PersistentClient(path=local_path + chroma_db)

# 컬렉션 로드
image_collection = client.get_collection("outfit_img_embeddings")
text_collection = client.get_collection("outfit_txt_embeddings")
products_collection = client.get_collection("products_metadatas")


# GraphState 정의
class GraphState(TypedDict):
    session_id: str
    user_input_text: str
    translated_text: str
    user_input_image: bytes
    user_intent: str
    input_type: str
    retrieved_metadata: dict
    chatbot_answer: str
    chatbot_intent: str
    location: str
    context: str
    season: str
    preferred_style: str
    user_age: int
    user_gender: str


def session(state: GraphState):
    # 세션 상태 초기화
    if "store" not in st.session_state:
        st.session_state["store"] = {}

    if "store" not in st.session_state:
        st.session_state["chat_intent_history"] = {}
    return state


# 히스토리 관리 함수들
def get_session_history(session_id):
    if session_id not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_id] = ChatMessageHistory()
    return st.session_state["store"][session_id]  # 해당 세션 ID에 대한 세션 기록 반환


def get_session_intent_history(session_id):
    if (
        session_id not in st.session_state["chat_intent_history"]
    ):  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["chat_intent_history"][session_id] = ChatMessageHistory()
    return st.session_state["chat_intent_history"][
        session_id
    ]  # 해당 세션 ID에 대한 세션 기록 반환


# 타입 체크 함수
def typeCheck(state: GraphState):
    if state["user_input_text"] and state["user_input_image"]:
        state["input_type"] = "멀티"
    elif state["user_input_text"]:
        state["input_type"] = "텍스트"
    elif state["user_input_image"]:
        state["input_type"] = "이미지"
    return state


# 텍스트 번역 함수
def trans_user_input(state: GraphState):
    user_input = state["user_input_text"]
    translator = GoogleTranslator(source="auto", target="en")
    translated_text = translator.translate(user_input)
    state["translated_text"] = translated_text
    return state


# 의도 분류 함수
def intent_classification(state: GraphState):
    intent_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
    )

    # ChatPromptTemplate을 사용하여 메시지 템플릿 생성
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 사용자의 의도를 분류하는 챗봇입니다."),
            ("system", "사용자의 의도의 종류는 3가지로 구분합니다."),
            (
                "system",
                "대화 내용을 참고하여 대화의 전체적인 분위기로 의도를 분류하도록 합니다.",
            ),
            (
                "system",
                """
            # 사용자 의도의 종류
            [추천, 구매, 일반]
            """,
            ),
            ("system", "출력 형식은 단순히 의도만 출력합니다."),
            MessagesPlaceholder(
                variable_name="chat_history"
            ),  # 세션 기록을 담을 자리 표시자
            ("user", "{user_input}"),  # 사용자의 현재 입력을 처리
        ]
    )

    # RunnableWithMessageHistory를 사용하여 세션 기록을 처리
    intent_chain = template | intent_llm
    chain_with_history = RunnableWithMessageHistory(
        intent_chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="user_input",  # 사용자의 입력 키
        history_messages_key="chat_history",  # 세션 기록 키
    )

    # 최종적으로 의도를 분류하고 결과를 상태에 저장
    state["user_intent"] = chain_with_history.invoke(
        {"user_input": state["user_input_text"], "session_id": state["session_id"]}
    ).content

    return state


# 멀티모달 추천 함수
def multi_recommend(state: GraphState):
    user_image_path = state["user_input_image"]
    text_input = state["user_input_text"]

    # trans_user_input 함수에서 생성된 translated_text를 state에서 가져옴
    translated_text = state["translated_text"]

    # 이미지 열기
    image = PILImage.open(BytesIO(user_image_path)).convert("RGB")

    # 이미지 임베딩 생성
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs).cpu().numpy()

    # 임베딩 정규화
    image_embedding = image_embedding / np.linalg.norm(
        image_embedding, ord=2, axis=-1, keepdims=True
    )

    # Chroma DB에서 상위 30개의 유사한 데이터를 검색
    results = image_collection.query(
        query_embeddings=image_embedding.tolist(),  # 쿼리로 사용할 임베딩
        n_results=30,  # 상위 30개의 결과를 가져옴
        include=[
            "embeddings",
            "metadatas",
            "distances",
        ],  # 반환할 정보: 임베딩, 메타데이터, 거리
    )

    if not results["metadatas"] or len(results["metadatas"]) == 0:
        print("No results found.")
        return state

    # 번역된 텍스트로 텍스트 임베딩 생성
    text_inputs = processor(text=[translated_text], return_tensors="pt").to(device)
    with torch.no_grad():
        text_embedding = model.get_text_features(**text_inputs).cpu().numpy()

    # 텍스트 임베딩 정규화
    text_embedding = text_embedding / np.linalg.norm(
        text_embedding, ord=2, axis=-1, keepdims=True
    )

    # 텍스트와 상위 30개 이미지의 임베딩 간의 유사도 계산
    similarities = np.dot(np.array(results["embeddings"][0]), text_embedding.T)

    # 유사도가 가장 높은 이미지 선택
    best_match_idx = np.argmax(similarities)

    # 유효한 인덱스인지 확인
    if best_match_idx >= len(results["metadatas"][0]):
        print("Index out of range. Something went wrong.")
        return state

    # 최종 결과를 state에 저장
    state["retrieved_metadata"] = results["metadatas"][0][best_match_idx]
    state["retrieved_distance"] = similarities[best_match_idx]

    return state


def image_recommend(state: GraphState):
    user_image_path = state["user_input_image"]

    # 이미지 열기
    image = PILImage.open(BytesIO(user_image_path)).convert("RGB")

    # 이미지 임베딩 생성
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs).cpu().numpy()

    # 임베딩 정규화
    image_embedding = image_embedding / np.linalg.norm(
        image_embedding, ord=2, axis=-1, keepdims=True
    )

    # Chroma DB에서 상위 1개의 유사한 데이터를 검색
    results = image_collection.query(
        query_embeddings=image_embedding.tolist(),  # 쿼리로 사용할 임베딩
        n_results=1,  # 상위 1개의 결과를 가져옴
        include=["metadatas", "distances"],  # 반환할 정보: 메타데이터, 거리, IDs
    )

    if not results["metadatas"] or len(results["metadatas"]) == 0:
        print("No results found.")
        return state

    # 검색 결과를 처리하여 state에 저장
    state["retrieved_metadata"] = results["metadatas"][0]
    state["retrieved_distance"] = results["distances"][0]

    return state


# 추천 함수
def recommend(state: GraphState):
    translated_text = state["translated_text"]

    # 텍스트 임베딩 생성
    inputs = processor(
        text=[translated_text], return_tensors="pt", padding=True, truncation=True
    ).to(device)
    with torch.no_grad():
        text_embeddings = model.get_text_features(**inputs).cpu().numpy()

    # 임베딩 정규화
    text_embeddings = text_embeddings / np.linalg.norm(
        text_embeddings, ord=2, axis=-1, keepdims=True
    )

    # Chroma DB에서 상위 1개의 유사한 데이터 검색
    results = image_collection.query(
        query_embeddings=text_embeddings.tolist(),
        n_results=1,
        include=["metadatas", "distances"],
    )

    if not results["metadatas"] or len(results["metadatas"]) == 0:
        print("No results found.")
        return state

    # 검색 결과 저장
    state["retrieved_metadata"] = results["metadatas"][0]
    state["retrieved_distance"] = results["distances"][0]
    state["retrieved_translated_text"] = translated_text
    return state


# 챗봇 응답 생성 함수
def chatbot_output(state: GraphState):
    output_llm = ChatOpenAI(model="gpt-4o", temperature=0)

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "사용자의 입력에 대해서 사용자 의도를 고려하여 조언해주는 Assistant 챗봇입니다.",
            ),
            (
                "system",
                "사용자의 입력과 사용자 이전 입력 그리고 챗봇의 이전 응답을 참고하여 꼬리 질문을 하고, 이를 통해 사용자의 현재 상황을 알아냅니다.",
            ),
            ("system", "사용자에게 자연스럽게 구매를 이끌어내세요."),
            (
                "system",
                "이때 꼬리 질문은 한 번에 하나씩 하도록 하고 멀티턴 대화를 통해 정보를 획득하세요.",
            ),
            ("system", "5번의 멀티턴 대화 안에 추천을 진행하세요."),
            (
                "system",
                "알아내면 좋은 사용자의 상황:\n- 어디에 가서 입을 옷인가요?\n- 어떤 상황에서 입을 옷인가요?\n- 현재 날씨는 어떤가요?",
            ),
            MessagesPlaceholder(
                variable_name="chat_history"
            ),  # 세션 기록을 담을 자리 표시자
            ("user", "사용자 입력: {user_input}"),
            ("assistant", "검색된 메타데이터: {retrieved_metadata}"),
        ]
    )

    # RunnableWithMessageHistory를 사용하여 세션 기록을 처리
    intent_chain = template | output_llm
    chain_with_history = RunnableWithMessageHistory(
        intent_chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="user_input",  # 사용자의 입력 키
        history_messages_key="chat_history",  # 세션 기록 키
    )

    # 최종적으로 GPT 모델을 사용해 응답 생성
    state["chatbot_answer"] = chain_with_history.invoke(
        {
            "user_input": state["user_input_text"],
            "retrieved_metadata": state["retrieved_metadata"],
            "session_id": state["session_id"],
        }
    ).content

    return state


def session(state: GraphState):
    # 세션 상태 초기화
    if "store" not in st.session_state:
        st.session_state["store"] = {}

    return state


# 워크플로우 설정
workflow = StateGraph(GraphState)

# session 함수를 랭 그래프 노드로 추가
workflow.add_node(session)

# 기존 노드 추가
workflow.add_node(trans_user_input)
workflow.add_node(intent_classification)
workflow.add_node(recommend)
workflow.add_node(image_recommend)
workflow.add_node(multi_recommend)
workflow.add_node(chatbot_output)
workflow.add_node(typeCheck)

# session 함수를 typeCheck 노드 앞에 연결
workflow.add_edge("session", "typeCheck")

workflow.add_conditional_edges(
    "typeCheck",
    lambda state: state["input_type"],
    {
        "텍스트": "trans_user_input",  # 텍스트만 입력되면 trans_user_input을 실행합니다.
        "이미지": "image_recommend",  # 이미지만 입력되면 image_recommend를 실행합니다.
        "멀티": "trans_user_input",  # 멀티모달 입력(텍스트 + 이미지)이면 trans_user_input을 실행합니다.
    },
)

workflow.add_conditional_edges(
    "trans_user_input",
    lambda state: state["input_type"],
    {
        "텍스트": "intent_classification",  # 텍스트가 주어졌을 때 intent_classification을 실행합니다.
        "멀티": "multi_recommend",  # 멀티모달 입력이 주어졌을 때 multi_recommend를 실행합니다.
    },
)

workflow.add_conditional_edges(
    "intent_classification",
    lambda state: state["user_intent"],
    {
        "추천": "recommend",  # 추천 의도라면 recommend를 실행합니다.
        "구매": "chatbot_output",  # 구매 의도라면 chatbot_output을 실행합니다.
        "일반": "chatbot_output",  # 일반 대화라면 chatbot_output을 실행합니다.
    },
)

workflow.add_edge("recommend", "chatbot_output")
workflow.add_edge("image_recommend", "chatbot_output")
workflow.add_edge("multi_recommend", "chatbot_output")

# 워크플로우의 시작 지점을 session으로 설정
workflow.set_entry_point("session")

workflow.set_finish_point("chatbot_output")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Streamlit UI 구성
st.title("Chatbot Ver3")

with st.sidebar:
    session_id = st.text_input("세션 ID를 입력하세요:", "default_session")
    user_input_image = st.file_uploader("이미지를 업로드하세요:")

user_input_text = st.text_input("텍스트로 질문하세요:")

if st.button("전송"):
    state = GraphState(
        session_id=session_id,
        user_input_text=user_input_text,
        translated_text=None,
        user_input_image=None,
        user_intent=None,
        input_type=None,
        retrieved_metadata={},
        chatbot_answer=None,
        preferred_style=None,
        user_age=None,
        user_gender=None,
    )

    if user_input_image:
        state["user_input_image"] = (
            user_input_image.read()
        )  # 이미지를 바이트로 읽어 저장

    if user_input_text:
        state = trans_user_input(state)

    config = RunnableConfig(
        recursion_limit=100,
        configurable={"session_id": session_id, "thread_id": "chatbot"},
    )

    state = app.invoke(state, config=config)

    # User's input displayed in chat format
    st.write(state["user_input_text"])

    # 이미지가 이미 표시된 적이 있는지 체크하기 위한 플래그
    image_displayed = False

    # Chatbot's response displayed in chat format
    st.write(state["chatbot_answer"])

    if state["retrieved_metadata"] and not image_displayed:
        first_item = state["retrieved_metadata"]
        image_url = (
            first_item.get("outfit_img_url") if isinstance(first_item, dict) else None
        )

        # 이미지와 설명을 출력
        if image_url:
            st.image(image_url, caption="추천 이미지")
            # st.write(f"이미지 설명: {state['chatbot_answer']}")
            image_displayed = True  # 이미지가 출력되었음을 기록
        else:
            st.write("이미지를 찾을 수 없습니다.")
    elif not state["retrieved_metadata"]:
        st.write("유사한 이미지를 찾을 수 없습니다.")
