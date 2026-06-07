import pandas as pd
import numpy as np
import time
from sqlalchemy import text
from sqlalchemy.orm import Session
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging

from services.vector_db_service import _get_chroma_client, _get_embedding_function, _get_collection

logger = logging.getLogger(__name__)

def run_clustering_job(db: Session):
    logger.info("Mulai proses clustering dari ChromaDB...")

    # 1. Fetch ChromaDB Data
    embedding_func = _get_embedding_function()
    client = _get_chroma_client()
    collection = _get_collection(client, embedding_func)

    chroma_data = collection.get(include=["embeddings", "metadatas"])
    if not chroma_data or not chroma_data['ids']:
        logger.warning("ChromaDB kosong, skip clustering.")
        return

    embeddings = np.array(chroma_data['embeddings'])
    metadatas = chroma_data['metadatas']
    
    # Buat DF sementara
    df_chroma = pd.DataFrame(metadatas)
    df_chroma['chroma_id'] = chroma_data['ids']
    
    # Pastikan data asset_id ada
    if 'asset_id' not in df_chroma.columns:
        logger.warning("Field 'asset_id' tidak ada di metadatas ChromaDB.")
        return

    # Karena metadata di ChromaDB sudah lengkap (termasuk asset_type, root_cause, spare_parts, repair_cost)
    # kita bisa langsung menggunakan df_chroma untuk clustering dan agregasi
    df_full = df_chroma.copy()
    
    # 2. Looping Clustering per Asset Type
    if 'asset_type' not in df_full.columns:
        logger.warning("Field 'asset_type' tidak ada di metadatas ChromaDB. Harap re-index data.")
        return

    unique_asset_types = df_full['asset_type'].dropna().unique()
    
    for asset_type in unique_asset_types:
        indices = df_full.index[df_full['asset_type'] == asset_type].tolist()
        df_filtered = df_full.iloc[indices].copy().reset_index(drop=True)
        embeddings_filtered = embeddings[indices]
        
        total_data = len(df_filtered)
        # KMeans membutuhkan n_samples >= n_clusters. Karena min_k default adalah 4,
        # kita butuh minimal 5 data agar pencarian range K (4 s/d total_data - 1) valid.
        if total_data < 5:
            logger.info(f"Asset Type '{asset_type}' skip clustering (Data < 5).")
            continue
            
        logger.info(f"Menganalisa '{asset_type}' dengan {total_data} data.")
        
        # PCA Dimensionality Reduction
        n_components = min(30, total_data - 1)
        if n_components > 5:
            pca = PCA(n_components=n_components, random_state=42)
            embeddings_for_clustering = pca.fit_transform(embeddings_filtered)
        else:
            embeddings_for_clustering = embeddings_filtered.copy()
            
        # Threshold Penentuan K optimal
        THRES_LOW = 1000
        THRES_MEDIUM = 6000
        
        if total_data <= THRES_LOW:
            min_k, max_k = 4, 8
        elif total_data <= THRES_MEDIUM:
            min_k, max_k = 8, 12
        else:
            min_k, max_k = 12, 16
            
        max_k = min(max_k, total_data - 1)
        k_range = range(min_k, max_k + 1)
        
        best_k = min_k
        best_score = -1
        
        for k in k_range:
            if k < 2: continue
            kmeans_eval = KMeans(n_clusters=k, random_state=42, n_init=3)
            labels_eval = kmeans_eval.fit_predict(embeddings_for_clustering)
            
            try:
                score = silhouette_score(
                    embeddings_for_clustering, labels_eval, metric='cosine', 
                    sample_size=min(2000, total_data), random_state=42
                )
            except ValueError:
                score = -1
                
            if score > best_score:
                best_score = score
                best_k = k
                
        # Final KMeans dengan best_k
        logger.info(f"K-Means Final untuk '{asset_type}' menggunakan K={best_k}")
        kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        final_labels = kmeans_final.fit_predict(embeddings_for_clustering)
        
        df_filtered['cluster'] = final_labels
        
        # 4. Insert ke clustering_runs (Sesi Run)
        run_insert_query = text("""
            INSERT INTO clustering_runs (asset_type, best_k, silhouette_score)
            VALUES (:asset_type, :best_k, :silhouette_score)
        """)
        db.execute(run_insert_query, {
            "asset_type": asset_type,
            "best_k": int(best_k),
            "silhouette_score": float(best_score)
        })
        run_id = db.execute(text("SELECT LAST_INSERT_ID()")).scalar()

        # 5. Loop Semua Cluster (K Cluster)
        def get_mode(series):
            m = series.dropna().mode()
            return str(m.iloc[0]) if not m.empty else "Tidak Diketahui"

        for i in range(best_k):
            df_cluster = df_filtered[df_filtered['cluster'] == i]
            if df_cluster.empty:
                continue

            # Agregasi per cluster
            dom_damage = get_mode(df_cluster['issue_type'])
            dom_cause = get_mode(df_cluster['root_cause'])
            dom_spare_part = get_mode(df_cluster['spare_parts_used'])
            
            cost_series = pd.Series(pd.to_numeric(df_cluster['repair_cost'], errors='coerce'))
            avg_cost = int(cost_series.mean()) if pd.notna(cost_series.mean()) else 0
            member_count = len(df_cluster)

            # 6. Insert ke nlp_clusters (Detail Cluster)
            cluster_insert_query = text("""
                INSERT INTO nlp_clusters (
                    run_id, cluster_index, dominant_damage, dominant_cause, 
                    dominant_spare_part, average_cost, member_count
                ) VALUES (
                    :run_id, :cluster_index, :dominant_damage, :dominant_cause,
                    :dominant_spare_part, :average_cost, :member_count
                )
            """)
            db.execute(cluster_insert_query, {
                "run_id": run_id,
                "cluster_index": i,
                "dominant_damage": dom_damage,
                "dominant_cause": dom_cause,
                "dominant_spare_part": dom_spare_part,
                "average_cost": avg_cost,
                "member_count": member_count
            })
            cluster_id = db.execute(text("SELECT LAST_INSERT_ID()")).scalar()

            # 7. Mapping ticket_id ke cluster_id (Batch Insert)
            mapping_data = []
            for _, row in df_cluster.iterrows():
                mapping_data.append({
                    "run_id": run_id,
                    "ticket_id": int(row['ticket_id']),
                    "cluster_id": cluster_id
                })
            
            if mapping_data:
                mapping_query = text("""
                    INSERT INTO maintenance_log_clusters (run_id, ticket_id, cluster_id)
                    VALUES (:run_id, :ticket_id, :cluster_id)
                """)
                db.execute(mapping_query, mapping_data)

    db.commit()
    logger.info("Selesai proses clustering dan agregasi untuk seluruh tipe asset.")
