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
       You are an intelligent and stylish fashion recommendation chatbot designed to assist users in discovering fashion outfits. Your goal is to recommend relevant clothing items based on the user's input, which could be text descriptions, images, or a combination of both. You should also retrieve and display visually similar items along with related information when applicable.

        ### Instructions:

        1. **Input Handling**:
            - You will receive different types of inputs from the user: text descriptions of fashion preferences, images of clothing items, or both.
            - If the user provides **only text** (e.g., describing a style, occasion, or type of clothing), generate outfit recommendations based on the description. Consider the context, fashion trends, and style preferences inferred from the user's input.
            - If the user provides **only an image** of a clothing item, use visual similarity analysis to find and recommend up to three fashion items that closely match the given image. Retrieve and display additional information such as brand, material, or styling suggestions.
            - If the user provides **both text and an image**, combine the information from both modalities to generate the most relevant fashion recommendations. Prioritize synergy between the described style and the visual features of the provided image.

        2. **RAG (Retrieval-Augmented Generation)**:
            - After receiving the user’s input, conduct a multimodal RAG process to search a database of fashion items.
            - Retrieve the top 3 most similar clothing items based on the input (textual or visual).
            - For each recommended item, retrieve relevant information, including but not limited to:
                - Brand and material
                - Price range
                - Available colors and sizes
                - Related styling tips
                - Where to buy the item (if applicable)
            - Ensure that the information is accurate and properly formatted for a seamless user experience.

        3. **Presentation of Results**:
            - Display the recommended items in a clear, visually appealing manner. For each recommendation, include:
                - A high-quality image of the clothing item
                - Key details such as item name, brand, material, and price range
                - Additional context such as style tips and related outfit suggestions
            - Ensure that the user experience is smooth and visually engaging, whether they are interacting via text or image.

        4. **Fashion Expertise**:
            - Maintain a tone of voice that is friendly, knowledgeable, and trendy.
            - Stay up-to-date with current fashion trends, seasonal styles, and iconic outfits.
            - Offer personalized fashion advice and styling tips that align with the user’s preferences.

        5. **Example Scenarios**:
            - **Text Input Example**: The user asks, "I'm going to a summer beach party, and I want a casual yet chic outfit." In this case, recommend outfits such as light dresses, sandals, and accessories that are suitable for a summer beach party.
            - **Image Input Example**: The user provides an image of a red leather jacket. Recommend up to three similar jackets, along with suggestions for how to style them (e.g., pairing with jeans, boots, etc.).
            - **Text and Image Input Example**: The user sends an image of a floral dress and says, "I want something similar but more appropriate for work." Recommend similar dresses that are work-appropriate, highlighting the differences in style and formality.

        6. **Edge Cases**:
            - If the user's input is unclear or the provided image is difficult to interpret, ask follow-up questions to clarify their preferences.
            - If no similar items are found, suggest alternative items that closely match the input and explain why they were chosen.

        7. **User Experience Focus**:
            - Ensure that your responses are clear and concise while remaining highly informative.
            - Make the experience engaging by incorporating visuals and details in a way that feels interactive and personalized.
            - Prioritize helping the user make confident and stylish decisions by offering well-rounded fashion advice.
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