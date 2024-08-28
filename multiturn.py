from urllib import response
from requests import session
import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os
from PIL import Image  # 이미지 처리를 위한 모듈 추가


# API KEY 정보로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 프로젝트 이름을 입력합니다.
#logging.langsmith("[Project] multimodal_multiturn_chatbot")

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

if "store" not in st.session_state:
    st.session_state["store"] = {}


# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("reset")
    
    # 이미지 업로드
    uploaded_file = st.file_uploader("👚 이미지 업로드", type=["jpg", "jpeg", "png"])
    
    # 모델 선택 메뉴 -> 다른 거 선택으로 활용
    #selected_model = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini"], index=0)
    #selected_model = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-4',temperature=0)

    # 세션 ID 를 지정하는 메뉴 -> 얘는 사용자 아이디 concat.sth 이 되면 좋을 것 같은데.. 시간 없겠지
    #session_id = set.text_input("세션 ID를 입력하세요.", user.getId()) > 이런 식?
    session_id = st.text_input("세션 ID를 입력하세요.", "session_0821_01")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


# 체인 생성
def create_chain(model_name="gpt-4o"):

    # # 프롬프트 정의1
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             """
    #             You are a sophisticated fashion stylist and recommendation expert chatbot. Your role is to analyze the text queries or fashion images provided by the user and recommend suitable fashion items based on that input. 
    #             You are capable of providing personalized fashion suggestions, analyzing styles, and helping users with fashion-related decisions. 
                
    #             Your expertise includes:
    #             1. **Fashion trend analysis**: You are knowledgeable about the latest fashion trends and can offer styling tips accordingly. You recommend fashion items based on current seasonal trends, events, or specific fashion styles.
    #             2. **Style recommendations**: You analyze images provided by the user to identify and recommend similar styles or items. You focus on colors, patterns, textures, and details to find similar products.
    #             3. **Personalized suggestions**: Based on the user's preferences, you offer tailored fashion recommendations, whether it’s for casual wear, business casual, party outfits, or any specific occasion.
    #             4. **Multimodal input handling**: You can process both text queries and fashion images. You can analyze the style from an image and respond to text queries while also offering recommendations based on combined text and image input.
    #             5. **Fashion item comparison**: You are capable of comparing fashion items, discussing the pros and cons, and suggesting alternatives. You can also recommend fashion items within a specified budget.
    #             6. **Seasonal styling suggestions**: You provide styling advice and outfit suggestions suitable for different seasons, such as summer or winter. Additionally, you can recommend outfits for specific places like vacation spots or business trips.
                
    #             You should always provide responses in a friendly and easily understandable manner, offering unique fashion recommendations that cater to the user's style. Your goal is to help users make better fashion choices, enhancing their shopping experience with insightful and enjoyable fashion recommendations.

    #             When given a text query, you should provide detailed and insightful answers. When provided with an image, you should analyze it and recommend fashion items or offer styling suggestions based on the image's characteristics.
    #             Where appropriate, provide additional fashion tips and guidance to assist the user in making informed fashion decisions.
    #             """,
    #         ),
    #         # chat_history 
    #         MessagesPlaceholder(variable_name="chat_history"),
    #         ("human", "#Question:\n{question}"),  # 사용자 입력을 변수로 사용
    #     ]
    # )
    
    # 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
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
            ),
            # chat_history 
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "#Question:\n{question}"),  # 사용자 입력을 변수로 사용
        ]
    )

    # llm 생성
    llm = ChatOpenAI(model_name="gpt-4o")

    # 일반 Chain 생성
    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )
    return chain_with_history


# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자 인풋
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지용 영역 설정
warning_msg = st.empty()

if "chain" not in st.session_state:
    st.session_state["chain"] = create_chain(model_name='gpt-4o')


# 만약에 사용자 입력이 들어오면...
if user_input:
    chain = st.session_state["chain"]
    if chain is not None:
        response = chain.stream(
            # 질문 입력
            {"question": user_input},
            # 세션 ID 기준으로 대화를 기록합니다.
            config={"configurable": {"session_id": session_id}},
        )

        # 사용자의 입력
        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

            # 대화기록을 저장한다.
            add_message("user", user_input)
            add_message("assistant", ai_answer)
    else:
        # 이미지를 업로드 하라는 경고 메시지 출력
        warning_msg.error("이미지를 업로드 해주세요.")
        

# 업로드된 이미지 처리 -> 변경으로 인한 주석 처리
# if uploaded_file:
#     # 이미지를 PIL 이미지로 읽어오기
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # 예시로 이미지를 저장 (실제 사용에 따라 변형 가능)
#     image_path = f".cache/files/{session_id}_uploaded_image.png"
#     image.save(image_path)

#     # 이미지를 처리하는 코드를 여기에 추가하면 됩니다.
#     # 예를 들어, 이미지 특징 추출 및 관련 아이템 추천 처리.

#     # 임시로 이미지를 기반으로 추천한 응답 추가
#     st.chat_message("assistant").write("이미지를 분석하고 있습니다...")
 
# 로딩 애니메이션? -> 다른 함수 호출 전에 호출해서 사용 가능   
# import time

# with st.spinner('Loading...'):
#     time.sleep(5)
# st.success('Done!')