import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_teddynote.models import MultiModal

from dotenv import load_dotenv
import os
#
#상태 요약: 
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
st.sidebar.write(f"py 파일명: `{current_file_name}`")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []
    # st.session_state['messages'] = [{"role": "assistant", 
    #                                     "content": "안녕하세요! \b Set the Looks\b 패션 가이드 챗봇입니다.\b 패션에 대해 궁금한 것이 있다면 저에게 자유롭게 질문해주세요!"}]

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 이미지 업로드
    uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    # Process 버튼 눌렀을 때 이벤트 정의
    #process = st.button("Process")
    
    system_prompt = st.text_area(
    "system", 
        """
        Your task is to classify the user's intent into one of the following categories: ["Fashion Recommendation", "Fashion Feedback", "General Conversation"].
        Based on the user's intent, generate a natural, friendly, and conversational response in Korean. Use emojis when appropriate, and do not show the classification process to the user. The user should only see the final response.

        - **Fashion Recommendation**: The user is asking for fashion advice, outfit suggestions, or styling recommendations. Examples include phrases like "무슨 옷을 입어야 할까요?", "이 드레스에 맞는 신발을 추천해줘.", "요즘 유행이 뭐냐", or "여름에 어울리는 스타일이 뭐가 있을까?".
        - **Fashion Feedback**: The user is providing feedback on a fashion item, discussing personal preferences, or commenting on a suggested style. Examples include phrases like "이 옷 어때", "친구들이랑 여행가는데 이런 복장 어떨까", or "장례식에서 입어도 돼?".
        - **General Conversation**: The user is engaging in casual conversation unrelated to fashion advice or feedback. This could be small talk, general questions, or any other non-fashion related inquiry. Examples include phrases like "오늘 날씨가 어때요?", "어제 영화 봤어?", "아 배고프다" or "지금 몇 시예요?".

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
            return "이 스타일은 정말 세련되면서도 유니크한 느낌을 줄 수 있어요! ✨ \n\n\
                    제가 추천드리는 유사한 패션은 이 스타일과 잘 어울리는 트렌디한 아이템들입니다. 👗👠 \n\
                    특히, 이 드레스는 다양한 액세서리와 함께 매치하면 더욱 돋보일 거예요. 😊 \n\n\
                    여름에도 입기 좋은 시원한 소재로 제작되었고, 데일리 룩으로도 완벽하게 어울릴 거예요. \n\
                    이 스타일로 시도해보시면 많은 사람들이 눈여겨보게 될 거예요! 👍"
        elif result == "Fashion Feedback":
            return "이 이미지에 있는 아이템은 정말 멋지네요! 🎉 \n\n\
                    이 스타일은 다양한 코디에 잘 어울릴 것 같아요. 😊 \n\
                    특히 이 아이템은 어떤 룩에도 포인트를 줄 수 있어서 활용도가 높을 것 같아요!"
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
user_input = st.chat_input("패션에 대해 궁금한 것이 있다면 저에게 자유롭게 질문해주세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 만약에 사용자 입력이 들어오면...
if user_input:
    # 사용자의 입력을 추가
    st.chat_message("user").write(user_input)

    if uploaded_file:
        # 리스트의 마지막 이미지로 계속 대화하게 추가
        new_uploaded_file = uploaded_file[-1]  
        # 이미지가 있으면 이미지와 함께 처리
        image_filepath = process_imagefile(new_uploaded_file)
        
        # 이미지 시각화 추가
        with st.chat_message("user"):
            #st.image(image_filepath, caption="업로드된 이미지")
            st.image(image_filepath, caption=f"업로드된 이미지: {new_uploaded_file.name}")  # 파일명 추가

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
        

# 만약에 사용자 입력이 들어오면...
if user_input:
    # 사용자의 입력을 추가
    st.chat_message("user").write(user_input)

    if uploaded_file:
        new_uploaded_file = uploaded_file[-1]  
        # 이미지가 있으면 이미지와 함께 처리
        image_filepath = process_imagefile(new_uploaded_file)
        
        # 이미지 시각화 추가
        with st.chat_message("user"):
            st.image(image_filepath, caption="업로드된 이미지")

        response = generate_multimodal_answer(image_filepath, system_prompt, user_input, 'gpt-4o')

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
        text_answer = generate_text_answer(system_prompt, user_input, 'gpt-4')

        with st.chat_message("assistant"):
            st.write(text_answer)

        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", text_answer)
        
import openai
import numpy as np

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
    if user_input and uploaded_file[-1]:
        image_filepath = process_imagefile(uploaded_file[-1])
        
        # 텍스트와 이미지 기반으로 응답 생성
        response = embed_text_and_image(user_input, uploaded_file[-1], fclip_model)
        
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
        image_filepath = process_imagefile(uploaded_file[-1])

        # 이미지 기반으로 응답 생성
        response = embed_image(uploaded_file[-1], fclip_model)

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
        image_filepath = process_imagefile(uploaded_file[-1])

        # 이미지 기반으로 응답 생성
        response = embed_image(uploaded_file[-1], fclip_model)

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

import chromadb
import os

local_path = '/Users/jei/Downloads/'  
chroma_db = 'chromadb_j_0816'

# Chroma 클라이언트 생성
client = chromadb.PersistentClient(path=local_path+chroma_db)

# 모든 컬렉션 목록 가져오기
collections = client.list_collections()

# 컬렉션 정보 출력
for collection in collections:
    print(f"Collection name: {collection.name}")
    print("-" * 40)
    
# 저장된 ChromaDB 컬렉션 로드
img_collection = client.get_collection("outfit_img_embeddings")
txt_collection = client.get_collection("outfit_txt_embeddings")
products_collection = client.get_collection("products_metadatas")

# 데이터 수 확인해보기
print(f"img_collection 데이터 수 >> : {img_collection.count()}")
print(f"txt_collection  데이터 수 >>> : {txt_collection.count()}")
print(f"products_collection  데이터 수 >>> : {products_collection.count()}")
print("-" * 50)

