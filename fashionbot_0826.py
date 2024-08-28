import streamlit as st
from st_chat_message import message
from typing import TypedDict
from deep_translator import GoogleTranslator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
import torch
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from dotenv import load_dotenv
import numpy as np
import os
from langgraph.errors import GraphRecursionError
from langchain_core.runnables import RunnableConfig
import chromadb

local_path = "/Users/choibyounghoon/Desktop/test/"
chroma_db = "chromadb_j_0816"

# Chroma 클라이언트 생성
client = chromadb.PersistentClient(path=local_path + chroma_db)

# 저장된 ChromaDB 컬렉션 로드
image_collection = client.get_collection("outfit_img_embeddings")
text_collection = client.get_collection("outfit_txt_embeddings")
products_collection = client.get_collection("products_metadatas")

# 모든 컬렉션 목록 가져오기
collections = client.list_collections()

# 컬렉션 정보 출력
for collection in collections:
    print(f"Collection name: {collection.name}")
    print("-" * 40)

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
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit 제목 설정
st.title("대화 내용을 기억하는 패션 챗봇 💬")

# GraphState 정의
class GraphState(TypedDict):
    session_id: str
    user_input_text: str
    translated_text: str
    user_input_image: str
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


# 입력 유형 체크
def typeCheck(state: GraphState):
    if state["user_input_text"] is None:
        state["input_type"] = "이미지"
    else:
        state["input_type"] = "텍스트"
    return state


# 의도 분류 및 번역
def trans_user_input(state: GraphState):
    user_input = state["user_input_text"]
    # save_user_input_history(state["session_id"], user_input)

    translator = GoogleTranslator(source="auto", target="en")
    translated_text = translator.translate(user_input)
    state["translated_text"] = translated_text
    return state


def intent_classification(state: GraphState):
    intent_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    template = """
    당신은 패션 어시스턴트 챗봇입니다.
    {user_input} 에 대해 사용자의 의도를 파악해주세요.
    또한 구매 의도는 사용자의 이전 의도 기록이 추천이었을 경우에만 분류하도록 합니다.


    # 사용자 의도의 종류
    [추천, 구매, 일반]

    #출력 형식은 단순히 의도만 출력
    """

    #session_history = get_session_history(state["session_id"])
    #intent_history = get_chat_intent_history(state["session_id"])

    prompt = ChatPromptTemplate.from_template(template)
    intent_chain = prompt | intent_llm
    state["user_intent"] = intent_chain.invoke(
        {
            "user_input": state["user_input_text"],
        }
    ).content

    return state
    

# 추천 시스템
def recommend(state: GraphState):
    translator = GoogleTranslator(source="ko", target="en")
    translated_text = translator.translate(state["user_input_text"])
    state["translated_text"] = translated_text

    inputs = processor(
        text=[translated_text], return_tensors="pt", padding=True, truncation=True
    )
    with torch.no_grad():
        text_embeddings = model.get_text_features(**inputs).cpu().numpy()

    text_embeddings = text_embeddings / np.linalg.norm(
        text_embeddings, ord=2, axis=-1, keepdims=True
    )

    # 예제에서는 text_collection 이라는 외부 컬렉션에서 데이터를 조회한다고 가정
    results = image_collection.query(
        query_embeddings=text_embeddings.tolist(),
        n_results=1,
        include=["metadatas", "distances"],
    )

    a = results["metadatas"][0]
    b = a[0]
    c = b['outfit_img_url']
    state["retrieved_metadata"] = c
    
    return state


# 일반적인 응답 처리
def general(state: GraphState):
    return state


# ChatPromptTemplate을 사용한 챗봇 응답 생성
def chatbot_output(state: GraphState):
    output_llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # ChatPromptTemplate으로 프롬프트 정의
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
            ("user", "사용자 입력: {user_input}"),
            ("assistant", "검색된 메타데이터: {retrieved_metadata}"),
        ]
    )

    # st.session_state 데이터를 프롬프트에 포함
    #user_input_history = get_user_input_history(state["session_id"])
    #session_history = get_session_history(state["session_id"])
    #intent_history = get_chat_intent_history(state["session_id"])

    # ChatPromptTemplate을 이용한 체인 생성 및 실행
    chatbot_chain = template | output_llm

    state["chatbot_answer"] = chatbot_chain.invoke(
        {
            "user_input": state["user_input_text"],
            #"user_input_history": user_input_history,
            #"session_history": session_history,
            #"intent_history": intent_history,
            "retrieved_metadata": state["retrieved_metadata"],
        }
    ).content

    # 세션 기록에 저장
    #save_to_session_history(state["session_id"], state["chatbot_answer"])

    return state


# 챗봇 응답의 의도 파악
def output_check(state: GraphState):
    output_intent_llm = ChatOpenAI(model="gpt-4o")
    template = """
    당신은 챗봇의 답변이 어떤 목적으로 생성되었는지 알아내는 챗봇입니다.
    {chatbot_answer} 에 대해 해당 답변의 의도를 파악해주세요.

    #챗봇 답변 의도의 종류
    [추천 답변, 구매 답변, 꼬리 질문]

    #출력 형식은 단순히 의도만 출력
    """
    prompt = ChatPromptTemplate.from_template(template)
    intent_chain = prompt | output_intent_llm
    state["chatbot_intent"] = intent_chain.invoke(
        {"chatbot_answer": state["chatbot_answer"]}
    ).content
    return state


# 라우터 함수들
def router1(state: GraphState):
    return state["input_type"]


def router2(state: GraphState):
    return state["user_intent"]


def router3(state: GraphState):
    return state["chatbot_intent"]


# StateGraph 초기화
workflow = StateGraph(GraphState)

workflow.add_node(typeCheck)
workflow.add_node(intent_classification)
workflow.add_node(recommend)
workflow.add_node(general)
workflow.add_node(chatbot_output)
workflow.add_node(output_check)

# 노드 간의 연결 설정
workflow.add_conditional_edges(
    "typeCheck",
    router1,
    {
        "텍스트": "intent_classification",
        "이미지": "recommend",
    },
)

workflow.add_conditional_edges(
    "intent_classification",
    router2,
    {
        "추천": "recommend",
        "구매": "chatbot_output",
        "일반": "general",
    },
)

workflow.add_edge("recommend", "chatbot_output")
workflow.add_edge("general", "chatbot_output")
workflow.add_edge("chatbot_output", "output_check")

# 그래프의 시작점 및 끝점 설정
workflow.set_entry_point("typeCheck")
workflow.set_finish_point("output_check")

# 기록을 위한 메모리 저장소 설정
memory = MemorySaver()

# 그래프를 컴파일하여 앱 시작
app = workflow.compile(checkpointer=memory)

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
        chatbot_intent=None,
        location=None,
        context=None,
        season=None,
        preferred_style=None,
        user_age=None,
        user_gender=None,
    )

    if user_input_image:
        state["user_input_image"] = user_input_image

    if user_input_text:
        state = trans_user_input(state)

    
    config = RunnableConfig(recursion_limit=100, configurable={"thread_id": "chatbot"})

    state = app.invoke(state, config=config)

        # User's input displayed in chat format
    message(user_input_text, is_user=True)  # 사용자의 입력을 표시

    # Chatbot's response displayed in chat format
    message(state["chatbot_answer"], is_user=False)  # 챗봇의 응답을 표시

    # If there is an image to display
    if "retrieved_metadata" in state and state["retrieved_metadata"]:
        st.image(state["retrieved_metadata"], caption="추천 이미지")