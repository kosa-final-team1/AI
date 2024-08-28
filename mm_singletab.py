import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
import openai
from langchain_teddynote.models import MultiModal
from dotenv import load_dotenv
import os
from PIL import Image

import numpy as np

# API KEY 정보로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("[Error] OpenAI API Key를 찾을 수 없습니다.")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.set_page_config(page_title="AI Fashion Chatbot", page_icon="👗")
st.title(":gray[_mm_singletab_]_Set the Looks 👚🔎")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("reset")
 
    # 이미지 업로드
    uploaded_file = st.file_uploader("📸 패션 이미지를 등록하세요", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    
    # 모델 선택 메뉴
    #selected_model = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini"], index=0)
    selected_model = "gpt-4o"
    
    # 시스템 프롬프트 추가
    system_prompt = st.text_area(
    #     "system prompt",
    #     """
    #    You are an intelligent and stylish fashion recommendation chatbot designed to assist users in discovering fashion outfits. Your goal is to recommend relevant clothing items based on the user's input, which could be text descriptions, images, or a combination of both. You should also retrieve and display visually similar items along with related information when applicable.
    #     """,
        "syste prompt",
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

# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

# 이미지을 캐시에 저장하는 함수
@st.cache_resource(show_spinner="업로드한 이미지를 처리 중입니다...")
def process_imagefile(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    return file_path

# 체인 생성
def generate_answer(image_filepath, system_prompt, user_prompt, model_name="gpt-4o"):
    llm = ChatOpenAI(temperature=0, model_name=model_name)
    multimodal = MultiModal(llm, system_prompt=system_prompt, user_prompt=user_prompt)
    answer = multimodal.stream(image_filepath)
    return answer

# 초기화 버튼 click
if clear_btn:
    st.session_state["messages"] = []

# 대화 기록 출력
for chat_message in st.session_state["messages"]:
    st.chat_message(chat_message.role).write(chat_message.content)

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지용 빈 영역 설정
warning_msg = st.empty()

# 이미지가 업로드되었는지 확인
if uploaded_file:
    image_filepath = process_imagefile(uploaded_file)
    st.image(image_filepath, caption="Uploaded Image", use_column_width=True)

# # 만약에 사용자 입력이 들어오면...
# if user_input:
#     # [사용자 인풋이 이미지일 경우]
#     if uploaded_file:
#         image_filepath = process_imagefile(uploaded_file)
#         response = generate_answer(image_filepath, system_prompt, user_input, selected_model)

#         # 사용자의 입력
#         st.chat_message("user").write(user_input)

#         # AI의 응답 처리
#         with st.chat_message("assistant"):
#             container = st.empty()
#             ai_answer = ""
#             for token in response:
#                 ai_answer += token.content
#                 container.markdown(ai_answer)

#         # 대화기록을 저장
#         add_message("user", user_input)
#         add_message("assistant", ai_answer)
#     else:
#         # 이미지를 업로드 하라는 경고 메시지 출력
#         warning_msg.error("이미지를 업로드 해주세요.")
 
 # ********************************
 # [기존] 임베딩 함수 매핑
  # ********************************

def process_user_input(user_input, fclip_model):

    if isinstance(user_input, str):
        # 사용자가 텍스트를 입력한 경우
        return embed_text(user_input, fclip_model)

    elif isinstance(user_input, Image.Image):
        # 사용자가 이미지를 입력한 경우
        return embed_image(user_input, fclip_model)

    elif isinstance(user_input, tuple) and len(user_input) == 2:
        # 사용자가 텍스트와 이미지를 모두 입력한 경우
        user_text, user_image = user_input
        if isinstance(user_text, str) and isinstance(user_image, Image.Image):
            return embed_text_and_image(user_text, user_image, fclip_model)
        else:
            warning_msg.error("입력이 잘못되었습니다.")

            #raise ValueError("입력이 잘못되었습니다. 텍스트는 문자열이어야 하고, 이미지는 이미지 파일이어야 합니다. 다시 확인해주세요.")

    else:
        raise ValueError("입력이 잘못되었습니다. 텍스트, 이미지, 또는 텍스트와 이미지를 함께 입력해야 합니다. 올바른 형식으로 입력해 주세요.")

# -------------------------------------------------------------------------------
# [0822] 아래로 기존 함수 다 넣기
# -------------------------------------------------------------------------------
# - system_prompt 추가해서 전달하는 방향으로 가야할 듯

# 번역
def translate_text(text, target_lang="en"):
    try:
        # GPT-4 모델을 사용해 번역 수행
        prompt = f"Translate the following text to {target_lang}: {text}"
        
        # GPT-4 API 호출
        response = openai.Completion.create(
            engine="gpt-4",  # 사용할 모델 (gpt-3.5-turbo도 사용 가능)
            prompt=prompt,
            max_tokens=200,  # 번역된 텍스트의 최대 길이
            temperature=0.3  # 응답의 일관성을 위한 낮은 온도 설정
        )
        
        # GPT-4 응답에서 번역된 텍스트 추출
        translation = response.choices[0].text.strip()
        return translation
    
    except Exception as e:
        print(f"Error during translation: {e}")
        return text  # 번역에 실패한 경우 원본 텍스트를 반환
    
# 임베딩 모델 및 의도별 함수 mapping


# 텍스트 임베딩 후 정규화
def embed_text(text, model):
    #번역 fn 먼저 실행되게 추가
    translated_text = translate_text(text, target_lang="en")

    text_embedding = model.encode_text([translated_text], batch_size=32)
    text_embedding = text_embedding / np.linalg.norm(text_embedding, ord=2, axis=-1, keepdims=True)
    return text_embedding

# 이미지 임베딩 함수 후 정규화
def embed_image(image, model):
    image_embedding = model.encode_images([image], batch_size=32)
    image_embedding = image_embedding / np.linalg.norm(image_embedding, ord=2, axis=-1, keepdims=True)
    return image_embedding

# 텍스트 + 이미지 임베딩 후 정규화
def embed_text_and_image(user_text, user_image, fclip_model):
    text_embedding = embed_text(user_text, fclip_model)
    image_embedding = embed_image(user_image, fclip_model)
    return text_embedding, image_embedding
 
# ********************************
# 사용자 인풋 타입에 따른 호출
# ********************************

#pip install -U fashion_clip 
from fashion_clip.fashion_clip import FashionCLIP

fclip_model = FashionCLIP('fashion-clip')
if user_input or uploaded_file:
    # 이미지와 텍스트 둘 다 입력된 경우
    if user_input and uploaded_file:
        image_filepath = process_imagefile(uploaded_file)
        
        # 텍스트와 이미지 기반으로 응답 생성
        response = embed_text_and_image(user_input, uploaded_file, fclip_model)
        
        # 사용자의 입력 처리
        st.chat_message("user").write(f"텍스트: {user_input}, 이미지 업로드됨")

        # AI의 응답 처리
        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            for token in response:
                ai_answer += token.content
                container.markdown(ai_answer)

        # 대화기록 저장
        add_message("user", f"텍스트: {user_input}, 이미지 업로드됨")
        add_message("assistant", ai_answer)

    # [사용자 인풋이 이미지일 경우]
    elif uploaded_file:
        image_filepath = process_imagefile(uploaded_file)

        # 이미지 기반으로 응답 생성
        response = embed_image(uploaded_file, fclip_model)

        # 사용자의 입력 처리
        st.chat_message("user").write("이미지 업로드됨")

        # AI의 응답 처리
        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            for token in response:
                ai_answer += token.content
                container.markdown(ai_answer)

        # 대화기록 저장
        add_message("user", "이미지 업로드됨")
        add_message("assistant", ai_answer)

    # [사용자 인풋이 텍스트일 경우]
    elif user_input:
        # 텍스트 기반으로 응답 생성
        response = embed_text(user_input, fclip_model)

        # 사용자의 입력 처리
        st.chat_message("user").write(user_input)

        # AI의 응답 처리
        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            for token in response:
                ai_answer += token.content
                container.markdown(ai_answer)

        # 대화기록 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)        
        