import chromadb
import os

local_path = '/Users/jei/Downloads/'  
chroma_db = 'chromadb_j_0816'
# ChromaDB 클라이언트 생성
client = chromadb.PersistentClient(path=local_path+chroma_db)

# 컬렉션 가져오기
collection = client.get_collection("your_collection_name")

# 모든 데이터를 추출하기 (임베딩과 메타데이터)
all_data = collection.get(include=["embeddings", "metadatas", "ids"])
# 저장된 ChromaDB 컬렉션 로드
img_collection = client.get_collection("outfit_img_embeddings")
txt_collection = client.get_collection("outfit_txt_embeddings")
products_collection = client.get_collection("products_metadatas")

img_data = img_collection.include=["embeddings", "metadatas"]
txt_data = txt_collection.include=["embeddings", "metadatas"]
product_data = products_collection.include=["embeddings", "metadatas"]

# 데이터 수 확인해보기
print(f"img_collection 데이터 수 >> : {img_collection.count()}")
print(f"txt_collection  데이터 수 >>> : {txt_collection.count()}")
print(f"products_collection  데이터 수 >>> : {products_collection.count()}")
print("-" * 50)


# # 딕셔너리에 저장
# extracted_img_data = []

# for i in range(len(img_data["ids"])):
#     item = {
#         "id": img_data["ids"][i],
#         "embedding": img_data["embeddings"][i],
#         "metadata": img_data["metadatas"][i]
#     }
#     extracted_img_data.append(item)

# # 추출된 데이터 확인
# print(extracted_img_data)