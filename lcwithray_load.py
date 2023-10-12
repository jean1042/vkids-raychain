import time
import numpy as np
import ray
from langchain.vectorstores import FAISS
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

############## 1. Download the files locally for processing ##############
# path : /Users/pjiyoon/desktop/langchain/Preconfigured_vSphere_Alarms.html

############## 2. Database Index / Sharding configuration ##############
FAISS_INDEX_PATH = "faiss_index_fast" # Vector DB's index 지정
db_shards = 8

############## 3. Document Load ##############
loader = ReadTheDocsLoader("/Users/pjiyoon/desktop/langchain/Preconfigured_vSphere_Alarms.html")

############## 4. Split the Text ##############
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # maximum number of characters that a chunk can contain
    chunk_overlap=20,  # 인접한 두 chunck 사이에 겹쳐야 하는 문자 수, chunk 사이의 연속성을 유지하기 위해 약간 겹치는게 좋다고 함
    length_function=len,
)

############## 5. Shard processing function ##############
@ray.remote(num_gpus=1)
def process_shard(shard):
    print(f'Starting process_shard of {len(shard)} chunks')
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings("multi-qa-mpnet-base-dot-v1")
    result = FAISS.from_documents(shard, embeddings)
    end_time = time.time()-start_time
    print(f'Shard completed in {end_time} seconds')
    return result

################################## BEGIN PROCESS  ##################################
    # STAGE 1: Read all the docs, split them into chunks
    start_time = time.time()
    print("Loading documents...")
    docs = loader.load()
    chunks = text_splitter.create_documents(
        [doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs]
    )
    et = time.time() - start_time
    print(f"Time taken: {et} seconds.")

    # STAGE 2 : embed the docs
    embeddings = LocalHuggingFaceEmbeddings("multi-qa-mpnet-base-dot-v1")
    process_shard()
    print(f"Loading chunks into vector store ...")
    st = time.time()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(FAISS_INDEX_PATH)
    et = time.time() - st
    print(f"Time taken: {et} seconds.")