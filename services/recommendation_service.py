import datetime
import json
import time
import logging
from sqlalchemy import text
from sqlalchemy.orm import Session

from core.config import settings
from services.ml_service import predict_rul
from models.schemas import AssetInput
from services.vector_db_service import _get_chroma_client, _get_embedding_function, _get_collection

try:
    from groq import Groq
except ImportError:
    Groq = None

logger = logging.getLogger(__name__)

# Rule Thresholds as specified by the user
RUL_THRESHOLDS = {
    # Kelompok 1: Safety & Security
    "Sistem Pemadam Kebakaran": { "warning": 1.0, "critical": 0.25 },
    "Sistem Proteksi Kebakaran Aktif": { "warning": 1.0, "critical": 0.25 },
    "Security Sistem": { "warning": 1.0, "critical": 0.25 },

    # Kelompok 2: IT & Telecom
    "Sistem Telekomunikasi Gedung": { "warning": 2.0, "critical": 0.5 },
    "Pencatatan Meter": { "warning": 2.0, "critical": 0.5 },

    # Kelompok 3: Core Operations (M&E)
    "Mechanical": { "warning": 3.0, "critical": 1.0 },
    "Electrical": { "warning": 3.0, "critical": 1.0 },
    "Ventilasi Sistem": { "warning": 3.0, "critical": 1.0 },
    "Sistem Transportasi Gedung": { "warning": 3.0, "critical": 1.0 },
    "Sistem Energi": { "warning": 3.0, "critical": 1.0 },

    # Kelompok 4: Sipil & Plumbing
    "Civil": { "warning": 5.0, "critical": 2.0 },
    "Arsitektur": { "warning": 5.0, "critical": 2.0 },
    "Plumbing": { "warning": 5.0, "critical": 2.0 },
    "Distribusi Air": { "warning": 5.0, "critical": 2.0 },

    # Kelompok 5: Lainnya
    "Latihan Balakar": { "warning": 1.5, "critical": 0.5 }
}

def calculate_operating_hours(operational_hours_db: float, installation_date) -> float:
    """
    Hitung operating hours: operational hours di db * 5/7 * (tanggal hari ini - tanggal instalasi)
    """
    if not installation_date:
        return 0.0
    
    if isinstance(installation_date, datetime.datetime):
        installation_date = installation_date.date()
    elif isinstance(installation_date, str):
        try:
            installation_date = datetime.datetime.strptime(installation_date, "%Y-%m-%d").date()
        except ValueError:
            return 0.0

    today = datetime.date.today()
    delta_days = (today - installation_date).days
    delta_days = max(0, delta_days)
    
    op_hours_base = operational_hours_db if operational_hours_db is not None else 0.0
    return op_hours_base * (5.0 / 7.0) * delta_days

def get_asset_health(category: str, rul_value: float) -> str:
    """
    Fungsi penentu status kesehatan aset berdasarkan kategori dan nilai RUL
    """
    threshold = RUL_THRESHOLDS.get(category)
    if not threshold:
        return "Unknown"

    if rul_value <= threshold["critical"]:
        return "Critical"
    elif rul_value <= threshold["warning"]:
        return "Warning"
    else:
        return "Healthy"

def run_narrative_recommendation_job(db: Session, batch_limit: int = 15):
    """
    Cron job task untuk membuatkan rekomendasi narative bagi aset yang statusnya Aktif
    dan recommendation_narrative masih NULL (atau belum dibuat).
    """
    logger.info("Memulai job pembuatan rekomendasi narasi untuk aset Aktif...")
    
    if Groq is None:
        logger.error("Gagal menjalankan job: library 'groq' tidak terinstall.")
        return

    # 1. Inisialisasi client ChromaDB
    try:
        embedding_func = _get_embedding_function()
        chroma_client = _get_chroma_client()
        collection = _get_collection(chroma_client, embedding_func)
    except Exception as e:
        logger.error(f"Gagal koneksi ke ChromaDB: {e}")
        return

    # 2. Inisialisasi Groq client
    if not settings.GROQ_API_KEY:
        logger.error("Gagal menjalankan job: GROQ_API_KEY belum dikonfigurasi.")
        return
    groq_client = Groq(api_key=settings.GROQ_API_KEY)

    # 3. Ambil data aset yang berstatus 'Aktif' dan recommendation_narrative nya masih kosong (NULL)
    # Kami batasi per run (misal 15 aset) untuk menghindari Groq API Rate Limit.
    query_assets = text("""
        SELECT asset_id, asset_name, asset_brand, asset_model, category, sub_category, 
               asset_type, building, floor, zone, critical_level, instalation_date, 
               operational_hours, predicted_rul
        FROM assets
        WHERE status = 'Aktif' AND recommendation_narrative IS NULL
        LIMIT :limit
    """)
    
    active_assets = db.execute(query_assets, {"limit": batch_limit}).fetchall()
    
    if not active_assets:
        logger.info("Semua aset Aktif sudah memiliki rekomendasi narasi. Job selesai.")
        return

    logger.info(f"Ditemukan {len(active_assets)} aset Aktif yang perlu diproses pada batch ini.")

    for asset in active_assets:
        asset_id = asset.asset_id
        asset_name = asset.asset_name
        category = asset.category
        asset_type = asset.asset_type
        installation_date = asset.instalation_date
        operational_hours_db = asset.operational_hours
        predicted_rul_db = asset.predicted_rul
        
        logger.info(f"Memproses aset: {asset_name} (ID: {asset_id})")

        # A. Hitung Operating Hours
        calculated_op_hours = calculate_operating_hours(operational_hours_db, installation_date)
        
        # B. Tentukan RUL (Jika null, prediksi secara dinamis menggunakan ML service)
        rul_value = predicted_rul_db
        should_update_rul_db = False
        if rul_value is None:
            should_update_rul_db = True
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
                logger.info(f"RUL untuk {asset_name} diprediksi dinamis: {rul_value:.2f} tahun.")
            except Exception as e:
                logger.warning(f"Gagal memprediksi RUL secara dinamis untuk {asset_name}: {e}. Menggunakan default 30.0")
                rul_value = 30.0

        # C. Tentukan Status Kesehatan (Asset Health)
        health_status = get_asset_health(category, rul_value)

        # D. RAG STEP 1: Query riwayat di ChromaDB (Vector DB)
        search_query = f"Riwayat maintenance dan tindakan perbaikan untuk aset {asset_name} dengan status {health_status}"
        try:
            results = collection.query(
                query_texts=[search_query],
                n_results=3
            )
            # Tangkap dulu ke dalam variabel
            docs = results.get('documents')
            
            # Validasi variabelnya secara eksplisit
            if docs is not None and len(docs) > 0 and docs[0]:
                context_riwayat = " ".join(docs[0])
            else:
                context_riwayat = "Tidak ada riwayat insiden historis spesifik yang tercatat untuk aset ini."
        except Exception as e:
            logger.warning(f"Error querying ChromaDB untuk {asset_name}: {e}")
            context_riwayat = "Tidak ada riwayat insiden historis spesifik yang tercatat untuk aset ini."

        # E. RAG STEP 2: Ambil data nlp_clusters untuk asset_type ini (Modus Cluster Terbanyak)
        # Ambil cluster terbaru yang di-generate oleh clustering job
        query_cluster = text("""
            SELECT cluster_id, dominant_damage, dominant_cause, dominant_spare_part, estimated_cost
            FROM nlp_clusters
            WHERE asset_type = :asset_type
            ORDER BY last_clustered_at DESC, cluster_id DESC
            LIMIT 1
        """)
        cluster_info = db.execute(query_cluster, {"asset_type": asset_type}).fetchone()

        if cluster_info:
            similar_complaint_info = (
                f"Cluster {cluster_info.cluster_id} (Populasi Terbesar). "
                f"Karakteristik dominan: {cluster_info.dominant_damage} "
                f"yang disebabkan oleh {cluster_info.dominant_cause} "
                f"dengan rata-rata biaya perbaikan Rp{cluster_info.estimated_cost} "
                f"serta memerlukan penggantian {cluster_info.dominant_spare_part}."
            )
        else:
            similar_complaint_info = "Tidak ada rumpun kelompok kerusakan serupa untuk tipe aset ini."

        # F. Format data ML lengkap ke JSON (sebagai parameter Prompt)
        payload_final = {
            "id_aset_target": asset_name,
            "tipe_aset": asset_type,
            "rul": round(rul_value, 2),
            "asset_health": health_status,
            "operating_hours": int(calculated_op_hours),
            "Uptime": 100.0,  # default / ignored
            "MTBF": 0,       # default / ignored
            "MTTR": 0.0      # default / ignored
        }
        hasil_ml_json_string = json.dumps(payload_final, indent=4)

        # G. RAG STEP 3: Prompt Engineering (Sesuai persis dengan prompt di NLP.ipynb)
        prompt = f"""Anda adalah AI Modul Prediktif ASPEC (Smart Asset Management System).
Tugas Anda adalah menyusun dokumen rekomendasi tindakan maintenance preventif yang taktis dan konkret untuk tim teknisi di lapangan.

[DATA INPUT UTAMA]
- ID Aset Target: {asset_name}
- Status Kesehatan: {health_status}
- Sisa Masa Pakai Efektif (RUL): {round(rul_value, 2)} Tahun
- Detail Data ML Lengkap (JSON): {hasil_ml_json_string}
- Konteks Historis Riwayat Aset (Vector DB): {context_riwayat}
- Rumpun Kelompok Kerusakan Serupa (Hasil Clustering & Modus Data): {similar_complaint_info}

[INSTRUKSI STRUKTUR OUTPUT]
1. Tulis rekomendasi tindakan eksklusif dalam Bahasa Indonesia yang formal, taktis, tegas, dan langsung ke poin instruksi perbaikan (actionable).
2. Output HARUS BERUPA HANYA SATU kalimat instruksi tindakan SANGAT pendek dan to-the-point. Jangan ada kalimat pembuka, penutup, atau bullet point.
3. Langsung mulai dengan kata kerja tindakan preventif, tanpa mengulang informasi aset, status, RUL, biaya, atau metrik performa.
4. Rekomendasi tindakan WAJIB diselaraskan secara logis dengan karakteristik 'Rumpun Kelompok Kerusakan Serupa' (pola modus kerusakan, penyebab, dan spare part) serta 'Konteks Historis Riwayat Aset' yang disediakan.
5. Patuhi batasan data: JANGAN menyarankan tindakan spekulatif atau menambahkan komponen/prosedur yang tidak relevan dengan rumpun masalah pada data input.
6. JANGAN menggunakan kata ganti pertama kedua atau ketiga jawab secara objektif saja.
7. JANGAN menggunakan kalimat saran seperti sebaiknya, seharusnya, saya merekomendasikan ataupun yang lain, jawab saja to the point.
"""

        # H. Panggil Groq API
        try:
            response = groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            raw_content = response.choices[0].message.content
            # Pakai (raw_content or "") biar kalau None otomatis berubah jadi string kosong ""
            recommendation_narrative = (raw_content or "Gagal menghasilkan rekomendasi.").strip()
            # Bersihkan jika ada tanda kutip luar
            if recommendation_narrative.startswith('"') and recommendation_narrative.endswith('"'):
                recommendation_narrative = recommendation_narrative[1:-1].strip()
            
            logger.info(f"Rekomendasi sukses dibuat: {recommendation_narrative}")
            
            # I. Simpan / Update Aset ke MySQL DB
            if should_update_rul_db:
                update_query = text("""
                    UPDATE assets
                    SET predicted_rul = :predicted_rul,
                        recommendation_narrative = :recommendation_narrative,
                        last_ml_updated_at = NOW()
                    WHERE asset_id = :asset_id
                """)
                db.execute(update_query, {
                    "predicted_rul": rul_value,
                    "recommendation_narrative": recommendation_narrative,
                    "asset_id": asset_id
                })
            else:
                update_query = text("""
                    UPDATE assets
                    SET recommendation_narrative = :recommendation_narrative,
                        last_ml_updated_at = NOW()
                    WHERE asset_id = :asset_id
                """)
                db.execute(update_query, {
                    "recommendation_narrative": recommendation_narrative,
                    "asset_id": asset_id
                })
            
            db.commit()
            
            # Jeda sebentar (1 detik) untuk menghindari rate-limit API Groq
            time.sleep(1.0)

        except Exception as api_err:
            logger.error(f"Gagal memproses LLM untuk {asset_name}: {api_err}")
            db.rollback()
            continue

    logger.info("Job pembuatan rekomendasi narasi selesai untuk batch saat ini.")
