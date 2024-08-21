import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
#from langchain_teddynote import logging
from langchain_teddynote.models import MultiModal

from dotenv import load_dotenv
import os


# API KEY 정보로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
            raise ValueError("[Error] OpenAI API Key를 찾을 수 없습니다..")
        
# 프로젝트 이름을 입력합니다.
#logging.langsmith("[Project] kosa_final_chatbot")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.set_page_config(page_title="AI Fashion Chatbot", page_icon="👚")
st.title(":gray[_AI FashionBot_]_Set the Looks 👚🔎")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

# 탭을 생성
main_tab1, main_tab2 = st.tabs(["이미지", "대화내용"])


# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 이미지 업로드
    uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])

    # 모델 선택 메뉴
    selected_model = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini"], index=0)
    #selected_model = "gpt-4o"
    #selected_model = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-4o',temperature=0)

    # 시스템 프롬프트 추가
    system_prompt = st.text_area(
        "system prompt",
        """
        You are an advanced fashion stylist and personal shopper chatbot. Your role is to assist users in finding the perfect fashion items based on their preferences, occasion, and personal style. You can recommend clothing, accessories, and styling tips that match the user's needs, preferences, and current fashion trends.

        Your expertise includes:
        1. **Fashion trend analysis**: You are up-to-date on the latest fashion trends and can offer recommendations based on current seasonal styles, popular items, and runway inspirations.
        2. **Personalized recommendations**: Based on the user's input (such as style preferences, body type, favorite colors, and budget), you provide tailored fashion advice. You consider factors like the user’s lifestyle, the occasion (e.g., work, casual, party), and even the weather.
        3. **Styling suggestions**: You suggest how to style different outfits, including pairing clothing items, selecting accessories, and matching shoes or bags to complete the look. You also provide advice on color combinations, patterns, and fabrics.
        4. **Multimodal input handling**: You can understand and respond to both text queries and images. For example, if a user uploads an image of a specific outfit or clothing item, you analyze the image and suggest similar styles or items that match the user's preferences.
        5. **Fashion item comparison**: You can compare fashion items, discussing their pros and cons, and help users decide between different clothing options. You can also recommend alternatives within a specific budget.
        6. **Occasion-specific styling**: You can suggest outfits and accessories for different events or situations, such as business meetings, dates, casual outings, vacations, or formal events.

        Please always respond in a friendly, informative, and helpful manner. Aim to provide clear and actionable fashion recommendations. Your goal is to help the user feel confident and stylish in their fashion choices. 

        ---
        """,
        
        height=200,
    )

        # # User Query: 
        # {question}

        # # Additional Context: 
        # {context}

        # Provide thoughtful, detailed, and personalized fashion recommendations in response to the user's query, and where possible, suggest additional tips to enhance the overall outfit.
        
# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        main_tab2.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 이미지을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 이미지를 처리 중입니다...")
def process_imagefile(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    return file_path


# 체인 생성
def generate_answer(image_filepath, system_prompt, user_prompt, model_name="gpt-4o"):
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,
        model_name=model_name,  # 모델명
    )

    # 멀티모달 객체 생성 (예시로 MultiModal 사용)
    multimodal = MultiModal(llm, system_prompt=system_prompt, user_prompt=user_prompt)

    # 이미지 파일로부터 질의(스트림 방식)
    answer = multimodal.stream(image_filepath)
    return answer


# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = main_tab2.empty()

# 이미지가 업로드가 된다면...
if uploaded_file:
    # 이미지 파일을 처리
    image_filepath = process_imagefile(uploaded_file)
    main_tab1.image(image_filepath)

# 만약에 사용자 입력이 들어오면...
if user_input:
    # 파일이 업로드 되었는지 확인
    if uploaded_file:
        # 이미지 파일을 처리
        image_filepath = process_imagefile(uploaded_file)
        # 답변 요청
        response = generate_answer(
            image_filepath, system_prompt, user_input, selected_model
        )

        # 사용자의 입력
        main_tab2.chat_message("user").write(user_input)

        with main_tab2.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token.content
                container.markdown(ai_answer)

        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # 이미지를 업로드 하라는 경고 메시지 출력
        warning_msg.error("이미지를 업로드 해주세요.")