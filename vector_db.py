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

product_amekaji = products_collection.get(
                    where={
                    "outfit_category": {
                        "$eq": "amekaji"
                     }
                    }
                    )
print(product_amekaji)
print("-" * 50)
