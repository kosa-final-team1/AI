import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_teddynote.models import MultiModal

from dotenv import load_dotenv
import os
#
#상태 요약: 이미지 업로드 가능, 프롬프트 ㄱㅊ
#
# API KEY 정보로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("[Error] OpenAI API Key를 찾을 수 없습니다..")
        
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

# 현재 실행 중인 파일명을 Streamlit에 표시(얘 너무 py 돌아가면서 실행해서 헷갈려서 써놓기)
current_file_name = os.path.basename(__file__)
st.sidebar.write(f"py파일: `{current_file_name}`")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 이미지 업로드
    uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    
    # Process 버튼 눌렀을 때 이벤트 정의
    process = st.button("Process")
    
    # 모델 선택 메뉴
    #selected_model = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini"], index=0)

    # 시스템 프롬프트 추가 
    # -> 너무 자세하면 또 일반 의도에서 too strict answer?
    # system_prompt = st.text_area(
    #     "system prompt",
    #     """
    #     You are an intelligent and stylish fashion recommendation chatbot designed to assist users in discovering fashion outfits. Your goal is to recommend relevant clothing items based on the user's input, which could be text descriptions, images, or a combination of both. You should also retrieve and display visually similar items along with related information when applicable.

    #     ### Instructions:

    #     1. **Input Handling**:
    #         - You will receive different types of inputs from the user: text descriptions of fashion preferences, images of clothing items, or both.
    #         - If the user provides **only text** (e.g., describing a style, occasion, or type of clothing), generate outfit recommendations based on the description. Consider the context, fashion trends, and style preferences inferred from the user's input.
    #         - If the user provides **only an image** of a clothing item, use visual similarity analysis to find and recommend up to three fashion items that closely match the given image. Retrieve and display additional information such as brand, material, or styling suggestions.
    #         - If the user provides **both text and an image**, combine the information from both modalities to generate the most relevant fashion recommendations. Prioritize synergy between the described style and the visual features of the provided image.

    #     2. **RAG (Retrieval-Augmented Generation)**:
    #         - After receiving the user’s input, conduct a multimodal RAG process to search a database of fashion items.
    #         - Retrieve the top 3 most similar clothing items based on the input (textual or visual).
    #         - For each recommended item, retrieve relevant information, including but not limited to:
    #             - Brand and material
    #             - Price range
    #             - Available colors and sizes
    #             - Related styling tips
    #             - Where to buy the item (if applicable)
    #         - Ensure that the information is accurate and properly formatted for a seamless user experience.

    #     3. **Presentation of Results**:
    #         - Display the recommended items in a clear, visually appealing manner. For each recommendation, include:
    #             - A high-quality image of the clothing item
    #             - Key details such as item name, brand, material, and price range
    #             - Additional context such as style tips and related outfit suggestions
    #         - Ensure that the user experience is smooth and visually engaging, whether they are interacting via text or image.

    #     4. **Fashion Expertise**:
    #         - Maintain a tone of voice that is friendly, knowledgeable, and trendy.
    #         - Stay up-to-date with current fashion trends, seasonal styles, and iconic outfits.
    #         - Offer personalized fashion advice and styling tips that align with the user’s preferences.

    #     5. **Example Scenarios**:
    #         - **Text Input Example**: The user asks, "I'm going to a summer beach party, and I want a casual yet chic outfit." In this case, recommend outfits such as light dresses, sandals, and accessories that are suitable for a summer beach party.
    #         - **Image Input Example**: The user provides an image of a red leather jacket. Recommend up to three similar jackets, along with suggestions for how to style them (e.g., pairing with jeans, boots, etc.).
    #         - **Text and Image Input Example**: The user sends an image of a floral dress and says, "I want something similar but more appropriate for work." Recommend similar dresses that are work-appropriate, highlighting the differences in style and formality.

    #     6. **Edge Cases**:
    #         - If the user's input is unclear or the provided image is difficult to interpret, ask follow-up questions to clarify their preferences.
    #         - If no similar items are found, suggest alternative items that closely match the input and explain why they were chosen.

    #     7. **User Experience Focus**:
    #         - Ensure that your responses are clear and concise while remaining highly informative.
    #         - Make the experience engaging by incorporating visuals and details in a way that feels interactive and personalized.
    #         - Prioritize helping the user make confident and stylish decisions by offering well-rounded fashion advice.
    #     """,
    #     height=500,
    # )
    # -> 이미지 인식 못 한다고 구라침 
    # system_prompt = st.text_area(
    #   "system",
    #             """
    #             You are an intelligent and stylish fashion recommendation chatbot designed to assist users in discovering fashion outfits...
    #             """,
    #     height=100,
    # )
    # system_prompt = st.text_area(
    #   "system",
    #             """
    #             You are a highly knowledgeable and fashion-savvy recommendation chatbot. Your primary role is to assist users in discovering and selecting the perfect fashion outfits based on their input, which may include text descriptions of style preferences, images of clothing items, or a combination of both.

    #             - When a user provides **text input** (e.g., describing an event, occasion, or specific style they are looking for), generate outfit suggestions that match their description, considering fashion trends, seasons, and personal style preferences.

    #             - When a user uploads an **image** of a clothing item, analyze the visual features of the item and suggest up to three similar outfits or accessories that complement the uploaded item. Provide relevant details such as brand, material, style suggestions, and how the item can be incorporated into different outfits.

    #             - If the user provides **both text and an image**, combine the information from both sources to offer highly personalized fashion recommendations that synergize the described style with the visual elements of the provided image.

    #             Be concise, friendly, and always on-trend, while offering clear explanations and styling tips that help the user feel confident in their fashion choices.                
    #             """,
    #     height=100,
    # )
    system_prompt = st.text_area(
    "system", 
        """
        Your task is to classify the user's intent into one of the following categories: ["Fashion Recommendation", "Fashion Feedback", "General Conversation"].
        Based on the user's intent, generate a natural, friendly, and conversational response in Korean. Use emojis when appropriate, and do not show the classification process to the user. The user should only see the final response.

        - **Fashion Recommendation**: The user is asking for fashion advice, outfit suggestions, or styling recommendations. Examples include phrases like "무슨 옷을 입어야 할까요?", "이 드레스에 맞는 신발을 추천해 주세요.", or "여름에 어울리는 스타일이 뭐가 있을까요?".
        - **Fashion Feedback**: The user is providing feedback on a fashion item, discussing personal preferences, or commenting on a suggested style. Examples include phrases like "이 옷이 정말 마음에 들어요", "이 스타일은 별로예요", or "이 재킷은 너무 작아요".
        - **General Conversation**: The user is engaging in casual conversation unrelated to fashion advice or feedback. This could be small talk, general questions, or any other non-fashion related inquiry. Examples include phrases like "오늘 날씨가 어때요?", "어제 영화 봤어?", or "지금 몇 시예요?".

        Based on the classification, respond naturally without revealing the classification to the user.

        ====
        def classify(sentence: str) -> Literal["Fashion Recommendation", "Fashion Feedback", "General Conversation"]:
            Classify the user's sentence into one of the categories based on their intent.

            Args:
                sentence (str): The sentence that needs to be classified.

            Returns:
                Literal["Fashion Recommendation", "Fashion Feedback", "General Conversation"]: The classification of the sentence.
        ====
        # Generate a natural, friendly, conversational response in Korean based on the classification:
        if result == "Fashion Recommendation":
            return "이 스타일 어떠세요? ✨ 이 옷은 정말 트렌디하고, 당신에게 딱 맞을 거예요! 👗👠"
        elif result == "Fashion Feedback":
            return "피드백 감사합니다! 😊 조금 더 도와드릴 수 있을까요? 더 마음에 드는 스타일을 찾아드릴게요! 🔍"
        elif result == "General Conversation":
            return "그렇군요! 😄 오늘 하루도 멋진 하루 보내세요! ☀️"
        
        Do not reveal the classification to the user. Only show the final response based on the classification.
        ("human", "{human}")
            """,
        height=100,
    )
    
    
# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 이미지을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="👗 업로드한 이미지를 처리 중입니다...")
def process_imagefile(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    return file_path


# 텍스트 기반 답변 생성
def generate_text_answer(system_prompt, user_prompt, model_name):
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,
        model_name=model_name,  # 모델명
    )

    # GPT-4 모델을 사용해 텍스트 기반의 응답 생성
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt)
    ]
    response = llm(messages)
    
    return response.content  # AIMessage 객체의 content 속성을 반환


# 체인 생성 (이미지 기반 멀티모달 응답 생성)
def generate_multimodal_answer(image_filepath, system_prompt, user_prompt, model_name):
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
warning_msg = st.empty()

# 만약에 사용자 입력이 들어오면...
if user_input:
    # 사용자의 입력을 추가
    st.chat_message("user").write(user_input)

    if uploaded_file:
        # 이미지가 있으면 이미지와 함께 처리
        image_filepath = process_imagefile(uploaded_file)
        
        # 이미지 시각화 추가
        with st.chat_message("user"):
            #st.image(image_filepath, caption="업로드된 이미지")
            st.image(image_filepath, caption=f"업로드된 이미지: {uploaded_file.name}")  # 파일명 추가

        response = generate_multimodal_answer(image_filepath, system_prompt, user_input, "gpt-4o")

        with st.chat_message("assistant"):
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
        # 이미지가 없으면 텍스트 기반으로 처리
        text_answer = generate_text_answer(system_prompt, user_input, "gpt-4")

        with st.chat_message("assistant"):
            st.write(text_answer)

        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", text_answer)