import json
import logging
import asyncio
from typing import AsyncGenerator, Dict, Any, List
from datetime import datetime, timedelta
from sqlalchemy import text
from sqlalchemy.orm import Session

from core.config import settings
from models.schemas import ChatRequest, AssetInput
from services.ml_service import predict_rul
from services.recommendation_service import calculate_operating_hours, get_asset_health
from services.vector_db_service import _get_chroma_client, _get_embedding_function, _get_collection

try:
    from groq import AsyncGroq
except ImportError:
    AsyncGroq = None

logger = logging.getLogger(__name__)

# Mock function for Vector DB search as requested
def search_vector_db(query: str, n_results: int = 3) -> dict:
    """
    Mock function to simulate searching vector DB.
    """
    return {
        "results": [
            f"Mock result 1 for: '{query}'",
            f"Mock result 2 for: '{query}'"
        ]
    }

def _build_system_prompt(db: Session, asset_id: int) -> str:
    # 1. Get asset data
    query_asset = text("""
        SELECT asset_id, asset_name, asset_brand, asset_model, category, sub_category, 
               asset_type, building, floor, zone, critical_level, instalation_date, 
               operational_hours, predicted_rul, status
        FROM assets
        WHERE asset_id = :asset_id
    """)
    asset = db.execute(query_asset, {"asset_id": asset_id}).fetchone()
    
    if not asset:
        return f"Anda adalah Asisten Virtual. Aset dengan ID {asset_id} tidak ditemukan di database."

    asset_name = asset.asset_name
    asset_type = asset.asset_type
    category = asset.category
    
    # 2. ML Payload data
    calculated_op_hours = calculate_operating_hours(asset.operational_hours, asset.instalation_date)
    rul_value = asset.predicted_rul
    if rul_value is None:
        try:
            data_input_ml = {
                "Tipe": asset_type or "AC Split",
                "Lokasi Gedung": asset.building or "Gedung B",
                "Lokasi Lantai": str(asset.floor or 1),
                "Lokasi Zona": asset.zone or "Tengah",
                "Operating_Hours": calculated_op_hours,
                "Total komplain": 0,
                "Total biaya perbaikan": 0.0
            }
            asset_input = AssetInput(**data_input_ml)
            rul_value = predict_rul(asset_input)
        except Exception:
            rul_value = 30.0

    
    # Hitung Sisa RUL (dalam tahun dan bulan)
    dynamic_rul_years = rul_value
    rul_formatted = f"{round(rul_value, 2)} tahun" # Default jika tidak ada tgl instalasi
    
    if asset.instalation_date:
        install_date = asset.instalation_date
        if isinstance(install_date, str):
            try:
                install_date = datetime.strptime(install_date, "%Y-%m-%d").date()
            except ValueError:
                pass
        elif isinstance(install_date, datetime):
            install_date = install_date.date()
            
        if type(install_date) is not str:
            today = datetime.today().date()
            # Asumsi rul_value (predicted_rul) adalah total masa pakai (lifespan) dalam Tahun
            total_lifespan_days = rul_value * 365.25
            eol_date = install_date + timedelta(days=total_lifespan_days)
            remaining_days = (eol_date - today).days
            
            # Hitung sisa RUL aktual dalam tahun untuk penentuan health status
            dynamic_rul_years = max(0.0, remaining_days / 365.25)
            
            if remaining_days <= 0:
                rul_formatted = "Masa pakai sudah habis"
            else:
                rem_years = remaining_days // 365
                rem_months = (remaining_days % 365) // 30
                if rem_years > 0:
                    rul_formatted = f"{int(rem_years)} tahun {int(rem_months)} bulan"
                else:
                    rul_formatted = f"{int(rem_months)} bulan"

    health_status = get_asset_health(category, dynamic_rul_years)
    logger.info(f"Health status: {health_status} with remaining rul {dynamic_rul_years:.2f} (original lifespan {rul_value}) for category {category}")
    
    payload_ml = {
        "id_aset_target": asset_name,
        "tipe_aset": asset_type,
        "rul": rul_formatted,
        "asset_health": health_status,
        "operating_hours": int(calculated_op_hours),
    }
    
    # 3. RAG Data (Vector DB)
    try:
        embedding_func = _get_embedding_function()
        chroma_client = _get_chroma_client()
        collection = _get_collection(chroma_client, embedding_func)
        
        # 1. Bikin query teks yang langsung to the point (tanpa filler words)
        search_query = "kendala kerusakan maintenance perbaikan" 
        
        # 1a. Cek total data dengan tipe ini (hanya metadata filter, tanpa semantic search)
        metadata_match = collection.get(
            where={"asset_type": asset_type}
        )
        total_metadata_match = len(metadata_match.get('ids', [])) if metadata_match else 0
        
        # 2. Gunakan Metadata Filtering (Mencari persis Tipenya) dengan semantic search
        results = collection.query(
            query_texts=[search_query],
            n_results=3,
            where={"asset_type": asset_type}
        )    
        docs = results.get('documents')
        total_query_match = len(docs[0]) if docs is not None and len(docs) > 0 else 0
        
        # 3. Log hasil informasinya sesuai permintaan
        logger.info(f"[Vector DB RAG] Tipe Aset '{asset_type}': {total_metadata_match} data tersedia di metadata | {total_query_match} data diambil oleh query search.")
        
        if docs is not None and len(docs) > 0 and len(docs[0]) > 0:
            report_text = " ".join(docs[0])
        else:
            report_text = "Tidak ada riwayat insiden historis spesifik yang tercatat untuk aset ini."
    except Exception as e:
        logger.warning(f"Error querying ChromaDB: {e}")
        report_text = "Tidak ada riwayat insiden historis spesifik yang tercatat untuk aset ini."
        
    # 4. Clustering context
    query_cluster = text("""
        SELECT cluster_id, dominant_damage, dominant_cause, dominant_spare_part, estimated_cost
        FROM nlp_clusters
        WHERE asset_type = :asset_type
        ORDER BY last_clustered_at DESC, cluster_id DESC
        LIMIT 1
    """)
    cluster_info = db.execute(query_cluster, {"asset_type": asset_type}).fetchone()

    if cluster_info:
        clustering_context = (
            f"Cluster {cluster_info.cluster_id} (Populasi Terbesar). "
            f"Karakteristik jenis kerusakan dominan: {cluster_info.dominant_damage} "
            f"penyebab kerusakan dominan : {cluster_info.dominant_cause} "
            f"dengan rata-rata biaya perbaikan : Rp{cluster_info.estimated_cost} "
            f"serta penggantian spare part dominan : {cluster_info.dominant_spare_part}."
        )
    else:
        clustering_context = "Tidak ada rumpun kelompok kerusakan serupa untuk tipe aset ini."

    # 5. Build prompt
    system_prompt = f"""Anda adalah Asisten Virtual Pakar Khusus untuk PIC Manajemen Aset ASPEC (Smart Asset Management System).

[IDENTITAS ENTITLE]
- ID Aset Target: {asset_name}
- Tipe Aset: {asset_type}

SUMBER DATA REKOMENDASI TACTICAL (Hasil RAG):
{report_text}

METRIK PERFORMA ENGINE TERKINI (Payload ML):
{json.dumps(payload_ml, indent=2)}

KONTEKS RUMPUN KERUSAKAN (Modus Clustering): 
{clustering_context}

INSTRUKSI KERJA CHATBOT:
1. Jawab seluruh pertanyaan PIC Manajemen Aset secara taktis, objektif, dan hanya bersandar pada data teknis di atas.
2. Anda harus menyadari penuh bahwa tipe aset yang sedang ditangani adalah {asset_type}, gunakan pengetahuan domain teknis yang relevan dengan tipe aset tersebut dalam batasan data yang ada.
3. Jika pengguna menanyakan tindakan konkret atau langkah preventif, berikan jawaban langsung berdasarkan 'SUMBER DATA REKOMENDASI TACTICAL' tanpa menyebutkan sumbernya.
4. Jika pengguna menanyakan potensi penyebab atau jenis komponen yang perlu disiapkan, berikan jawaban langsung dengan menganalisis 'KONTEKS RUMPUN KERUSAKAN' tanpa menyebutkan sumber atau detail klaster.
5. Anda memiliki akses ke tool `search_vector_db`. Jika pengguna menanyakan *contoh aset serupa* atau *daftar aset* dalam sebuah cluster, gunakan tool ini dengan query yang merangkum 'Jenis Kerusakan', 'Penyebab', dan 'Spare Part Digunakan' dari 'KONTEKS RUMPUN KERUSAKAN'. Kemudian, ekstrak hasil pencarian dan sebutkan sebagai contoh (jangan sertakan id aset di respon).
6. Gunakan Bahasa Indonesia yang profesional, tegas, berorientasi pada engineering, namun tidak kaku.
7. Patuhi batasan visual: Dilarang keras menggunakan emoticon/emoji ataupun tulisan tebal (bold text) dalam format jawaban Anda sesuai standar log sistem.
8. Asas Akurasi: Jangan pernah membuat asumsi parameter fisik atau menyarankan tindakan di luar batas konteks yang diberikan (No Hallucination).
9. JANGAN mencoba memanggil tool apapun yang tidak secara eksplisit terdaftar dan diberikan kepada Anda. Jika sebuah pertanyaan tidak dapat dijawab dari 'SUMBER DATA REKOMENDASI TACTICAL', 'METRIK PERFORMA ENGINE TERKINI', 'KONTEKS RUMPUN KERUSAKAN', atau dengan `search_vector_db` tool, nyatakan bahwa Anda tidak memiliki informasi untuk menjawab pertanyaan tersebut dan TIDAK BOLEH menambahkan komentar atau spekulasi tambahan.
"""
    
    logger.info(f"=== SYSTEM PROMPT GENERATED UNTUK ASSET ID {asset_id} ===\n{system_prompt}\n========================================================")
    
    return system_prompt

def _apply_sliding_window(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not history:
        return history
    
    system_prompt_msg = history[0]
    rest_of_history = history[1:]
    sliced_history = rest_of_history[-6:]
    
    return [system_prompt_msg] + sliced_history

async def stream_chat_response(db: Session, request: ChatRequest) -> AsyncGenerator[str, None]:
    if AsyncGroq is None:
        yield 'data: {"type": "error", "content": "Error: AsyncGroq client library is not installed."}\n\n'
        return
        
    if not settings.GROQ_API_KEY:
        yield 'data: {"type": "error", "content": "Error: GROQ_API_KEY is not configured."}\n\n'
        return

    client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    
    chat_history = request.chat_history
    
    # 1. Initialize system prompt if history is empty
    if not chat_history:
        system_prompt = _build_system_prompt(db, request.asset_id)
        chat_history = [{"role": "system", "content": system_prompt}]
    
    # 2. Append user question
    chat_history.append({"role": "user", "content": request.user_query})
    
    # 3. Apply sliding window
    chat_history = _apply_sliding_window(chat_history)
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_vector_db",
                "description": "Mencari dokumen di koleksi ChromaDB yang paling mirip dengan query yang diberikan. Args: query (str): Query pencarian. n_results (int): Jumlah hasil yang ingin dikembalikan.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "n_results": {"type": "number"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    try:
        # STAGE 1: Tool Check
        response = await client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=chat_history,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Execute tool
                if function_name == "search_vector_db":
                    tool_output = search_vector_db(**function_args)
                else:
                    tool_output = {"error": "Unknown tool"}
                
                # Append tool call & result to history
                chat_history.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": function_name,
                                    "arguments": tool_call.function.arguments
                                }
                            }
                        ]
                    }
                )
                
                chat_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_output)
                    }
                )

        # STAGE 2: Streaming Answer
        stream_response = await client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=chat_history,
            stream=True
        )
        
        full_answer = ""
        async for chunk in stream_response:
            content = chunk.choices[0].delta.content
            if content:
                full_answer += content
                yield f'data: {json.dumps({"type": "token", "content": content})}\n\n'
                
        # Append final answer to history
        chat_history.append({"role": "assistant", "content": full_answer})
        
        # Slide window one last time before returning
        chat_history = _apply_sliding_window(chat_history)
        
        # Send done signal
        final_payload = {
            "type": "done",
            "answer": full_answer,
            "updated_history": chat_history
        }
        yield f'data: {json.dumps(final_payload)}\n\n'
        
    except Exception as e:
        logger.error(f"Error during chat generation: {e}")
        yield f'data: {json.dumps({"type": "error", "content": str(e)})}\n\n'
