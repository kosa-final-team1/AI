import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os
from PIL import Image
from datetime import datetime

# =====================================
# img 무시하는 상태
# =====================================


# API KEY 정보로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.set_page_config(page_title="AI Fashion Chatbot", page_icon="👚")
st.title(":gray[_AI FashionBot_]_Set the Looks 👚🔎")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "store" not in st.session_state:
    st.session_state["store"] = {}

# 세션 id 설정 - 일단 임의로 추가해둠
user_id = "user_001"
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
session_id = f"{user_id}_{current_time}"

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("reset")
    
    # 이미지 업로드
    uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
    
    # 세션 ID 출력 (디버깅 용도)
    st.write(f"세션 ID: {session_id}")

# 이전 대화 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 새로운 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in st.session_state["store"]:  
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]

# 체인 생성
def create_chain(model_name="gpt-4o"):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an intelligent and stylish fashion recommendation chatbot designed to assist users in discovering fashion outfits...
                """,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "#Question:\n{question}")
        ]
    )

    llm = ChatOpenAI(model_name="gpt-4o")
    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  
        input_messages_key="question",  
        history_messages_key="chat_history",  
    )
    return chain_with_history

# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 출력
print_messages()

# 사용자 인풋
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지용 영역 설정
warning_msg = st.empty()

# AI 체인 생성
if "chain" not in st.session_state:
    st.session_state["chain"] = create_chain(model_name='gpt-4o')

# 만약에 사용자 입력이 들어오면...
if user_input or uploaded_file:
    chain = st.session_state["chain"]

    # 입력 데이터 생성 (이미지, 텍스트 모두 포함)
    input_data = {"question": user_input if user_input else ""}

    # 이미지 처리
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image_path = f".cache/files/{session_id}_uploaded_image.png"
        image.save(image_path)

        st.chat_message("assistant").write("이미지를 분석하고 있습니다...")

        input_data["image_path"] = image_path

    # 디버깅을 위해 input_data를 출력
    st.write("input_data: ", input_data)

    # 만약 이미지와 질문이 모두 비어있지 않다면 AI 모델로 전송
    if user_input or uploaded_file:
        # 텍스트 기반으로 AI 응답 생성
        response = chain.stream(
            input_data,
            config={"configurable": {"session_id": session_id}},
        )

        # 사용자의 입력 출력
        if user_input:
            st.chat_message("user").write(user_input)

        # AI의 응답 처리
        with st.chat_message("assistant"):
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

            # 대화기록 저장
            if user_input:
                add_message("user", user_input)
                add_message("assistant", ai_answer)