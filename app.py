import os
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import faiss

import streamlit as st

# 환경 변수 설정
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 경로 설정
data_path = "./data"
module_path = "./modules"

# Gemini 설정
import google.generativeai as genai

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

# Gemini 모델 선택
model = genai.GenerativeModel("gemini-1.5-flash")

# CSV 파일 로드
csv_file_path = "JEJU_MCT_DATA_modified.csv"
df = pd.read_csv(os.path.join(data_path, csv_file_path))

# 최신 연월 데이터만 가져옴
df = df[df["기준연월"] == df["기준연월"].max()].reset_index(drop=True)

# Streamlit App UI 설정
st.set_page_config(page_title="🍊참신한 제주 맛집!")

with st.sidebar:
    st.title("🍊참신한! 제주 맛집")
    st.write("")

    st.subheader("언드레 가신디가?")
    time = st.selectbox("시간대 선택", ["아침", "점심", "오후", "저녁", "밤"], key="time", label_visibility="hidden")
    st.write("")

    st.subheader("어드레가 맘에 드신디가?")
    local_choice = st.radio("맛집 선택", ("제주도민 맛집", "관광객 맛집"), label_visibility="hidden")
    st.write("")

st.title("혼저 옵서예!👋")
st.subheader("군맛난 제주 밥집🧑‍🍳 추천해드릴게예")
st.write("")
st.write("#흑돼지 #갈치조림 #옥돔구이 #고사리해장국 #전복뚝배기 #한치물회 #빙떡 #오메기떡..🤤")
st.write("")

image_path = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRTHBMuNn2EZw3PzOHnLjDg_psyp-egZXcclWbiASta57PBiKwzpW5itBNms9VFU8UwEMQ&usqp=CAU"
image_html = f"""
<div style="display: flex; justify-content: center;">
    <img src="{image_path}" alt="centered image" width="50%">
</div>
"""
st.markdown(image_html, unsafe_allow_html=True)
st.write("")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "어드런 식당 찾으시쿠과?"}
    ]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "어드런 식당 찾으시쿠과?"}
    ]

st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

# RAG 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "jhgan/ko-sroberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name).to(device)

print(f"Device is {device}.")

# FAISS 인덱스 로드 함수
def load_faiss_index(index_path=os.path.join(module_path, "faiss_index.index")):
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        print(f"FAISS 인덱스가 {index_path}에서 로드되었습니다.")
        return index
    else:
        raise FileNotFoundError(f"{index_path} 파일이 존재하지 않습니다.")

# 텍스트 임베딩
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()

# 임베딩 로드
embeddings = np.load(os.path.join(module_path, "embeddings_array_file.npy"))

# 응답 생성 함수
def generate_response_with_faiss(
    question, df, embeddings, model, embed_text, time, local_choice,
    index_path=os.path.join(module_path, "faiss_index.index"), max_count=10, k=3, print_prompt=True
):
    # FAISS 인덱스 로드
    index = load_faiss_index(index_path)

    # 검색 쿼리 임베딩 생성
    query_embedding = embed_text(question).reshape(1, -1)

    # 가장 유사한 텍스트 검색 (3배수)
    distances, indices = index.search(query_embedding, k * 3)

    # 상위 k개의 데이터프레임 추출
    filtered_df = df.iloc[indices[0, :]].copy().reset_index(drop=True)

    # 시간대 필터링
    if time == "아침":
        filtered_df = filtered_df[filtered_df["영업시간"].apply(lambda x: any(hour in eval(x) for hour in range(5, 12)))]
    elif time == "점심":
        filtered_df = filtered_df[filtered_df["영업시간"].apply(lambda x: any(hour in eval(x) for hour in range(12, 14)))]
    elif time == "오후":
        filtered_df = filtered_df[filtered_df["영업시간"].apply(lambda x: any(hour in eval(x) for hour in range(14, 18)))]
    elif time == "저녁":
        filtered_df = filtered_df[filtered_df["영업시간"].apply(lambda x: any(hour in eval(x) for hour in range(18, 23)))]
    elif time == "밤":
        filtered_df = filtered_df[filtered_df["영업시간"].apply(lambda x: any(hour in eval(x) for hour in [23, 24, 1, 2, 3, 4]))]

    if filtered_df.empty:
        return f"현재 선택하신 시간대({time})에는 영업하는 가게가 없습니다."

    filtered_df = filtered_df.head(k)

    # 현지인 맛집 옵션 반영
    local_choice = "제주도민(현지인) 맛집" if local_choice == "제주도민 맛집" else "현지인 비중이 낮은 관광객 맛집"

    if filtered_df.empty:
        return "질문과 일치하는 가게가 없습니다."

    reference_info = "\n".join(filtered_df["text"])
    
    # 시스템 메시지로 최대 토큰 수에 대한 힌트 추가
    # system_message = "Max Token은 5000으로 설정해주세요.\n"
    # prompt = f"{system_message}질문: {question} 특히 {local_choice}을 선호해\n참고할 정보:\n{reference_info}\n응답:"
    prompt = f"질문: {question} 특히 {local_choice}을 선호해\n참고할 정보:\n{reference_info}\n응답:"

    if print_prompt:
        print("-" * 90)
        print(prompt)
        print("-" * 90)
    
    response = model.generate_content(prompt)
    return response

# 사용자 입력 처리
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response_with_faiss(
                prompt, df, embeddings, model, embed_text, time, local_choice
            )
            placeholder = st.empty()
            full_response = response if isinstance(response, str) else response.text
            placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
