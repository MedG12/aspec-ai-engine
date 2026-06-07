import os
import time
from unittest import result
from sqlalchemy import text
from sqlalchemy.orm import Session
import chromadb
from chromadb.api import ClientAPI
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from core.config import settings
from models.schemas import BatchInsertRequest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import CursorResult # Import tipe datanya
from typing import cast #

# --- Constants ---
SBERT_MODEL_PATH = os.path.join("ml_models", "SBERT Model")
CHROMA_DB_PATH = settings.CHROMA_DB_PATH
COLLECTION_NAME = settings.CHROMA_COLLECTION_NAME
BATCH_SIZE = 1000

def _get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    """
    Load finetuned SBERT model sebagai embedding function untuk ChromaDB.
    """
    return SentenceTransformerEmbeddingFunction(model_name=SBERT_MODEL_PATH)

def _get_chroma_client() -> ClientAPI:
    """
    Inisialisasi ChromaDB PersistentClient ke path lokal.
    """
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)

def _get_collection(client: ClientAPI, embedding_func):
    """
    Get or create collection di ChromaDB.
    """
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )

def batch_insert_and_embed(db: Session, request: BatchInsertRequest) -> dict:
    """
    1. Insert data batch ke tabel MySQL maintenance_logs (is_embedded = FALSE).
    2. Embedding data ke Chroma DB.
    3. Update data di MySQL menjadi is_embedded = TRUE jika sukses.
    """
    if not request.records:
        return {
            "mode": "batch_insert",
            "indexed": 0,
            "total_in_db": get_collection_count(),
            "message": "Tidak ada data yang dikirim untuk di-insert."
        }

    # Kumpulkan ticket_ids dari request
    inserted_ids = [record.ticket_id for record in request.records]

    # 1. Embed ke ChromaDB
    embedding_func = _get_embedding_function()
    client = _get_chroma_client()
    collection = _get_collection(client, embedding_func)

    documents, metadatas, ids = [], [], []
    for record in request.records:
        ticket_id = record.ticket_id
        
        # Teks embedding difokuskan pada jenis kerusakan dan penyebab agar clustering lebih akurat
        # Kita tidak menyertakan biaya dan spare part di sini agar tidak menjadi noise saat clustering
        emb_parts = [
            f"Jenis: {record.issue_type or ''}",
            f"Penyebab: {record.root_cause or ''}"
        ]
        # Skip field yang kosong untuk hasil embedding yang lebih bersih
        text_doc = " | ".join([p for p in emb_parts if not p.strip().endswith(':')])
        
        # Build metadata untuk vector DB
        meta = {
            "ticket_id": str(ticket_id),
            "asset_id": str(record.asset_id),
            "asset_type": str(record.asset_type),
            "issue_type": str(record.issue_type or ''),
            "root_cause": str(record.root_cause or ''),
            "spare_parts_used": str(record.spare_parts_used or ''),
            "repair_cost": float(record.repair_cost) if record.repair_cost is not None else 0.0,
            "label": str(record.issue_type or 'Unknown')
        }
        
        documents.append(text_doc)
        metadatas.append(meta)
        ids.append(f"ticket_{ticket_id}")

    # Add to chroma DB dalam batch
    total_docs = len(documents)
    start_time = time.time()
    for i in range(0, total_docs, BATCH_SIZE):
        end_idx = min(i + BATCH_SIZE, total_docs)
        collection.upsert(
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )
        if i % 5000 == 0:
            elapsed = time.time() - start_time
            print(f"[VectorDB] Progress: {end_idx}/{total_docs} | {elapsed:.1f}s")

    # 2. Update MySQL is_embedded = TRUE
    if inserted_ids:
            id_list_str = ",".join(map(str, inserted_ids))
            update_query = text(f"UPDATE maintenance_logs SET is_embedded = TRUE WHERE ticket_id IN ({id_list_str})")
            
            try:
                # 1. Eksekusi query
                raw_result = db.execute(update_query)
                
                # 2. Beri tahu IDE (Type Checker) bahwa ini adalah CursorResult
                execute_result = cast(CursorResult, raw_result)
                
                # Sekarang IDE tidak akan protes soal .rowcount
                if execute_result.rowcount == 0:
                    print(f"[MySQL] Peringatan: Tidak ada data yang di-update. Ticket ID {inserted_ids} tidak ditemukan.")
                else:
                    print(f"[MySQL] Sukses: Berhasil meng-update {execute_result.rowcount} baris data.")
                
                db.commit()

            except SQLAlchemyError as e:
                db.rollback()
                print(f"[MySQL] Error gagal melakukan update: {str(e)}")
            except Exception as e:
                print(f"[Sistem] Terjadi kesalahan tak terduga: {str(e)}")
        
    final_count = collection.count()
    return {
        "mode": "batch_insert",
        "indexed": total_docs,
        "total_in_db": final_count,
        "message": f"Berhasil insert dan embedding {total_docs} data ke ChromaDB."
    }

def get_collection_count() -> int:
    """
    Return jumlah dokumen yang ada di ChromaDB collection.
    Return 0 jika collection belum ada atau terjadi error.
    """
    try:
        embedding_func = _get_embedding_function()
        client = _get_chroma_client()
        collection = _get_collection(client, embedding_func)
        return collection.count()
    except Exception:
        return 0
