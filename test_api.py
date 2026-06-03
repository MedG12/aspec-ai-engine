import requests
import json
from datetime import datetime
from sqlalchemy import text
from core.database import SessionLocal

url = "http://localhost:8000/api/predict-rul"

def get_all_active_payloads():
    db = SessionLocal()
    payloads = []
    try:
        # Ambil semua data aset aktif beserta agregasi komplain dan biaya perbaikannya sekaligus
        query = text("""
            SELECT 
                a.asset_id, 
                a.asset_type, 
                a.building, 
                a.floor, 
                a.zone, 
                a.instalation_date, 
                a.operational_hours,
                COUNT(m.ticket_id) as total_komplain,
                COALESCE(SUM(m.repair_cost), 0) as total_biaya_perbaikan
            FROM assets a
            LEFT JOIN maintenance_logs m ON a.asset_id = m.asset_id
            WHERE a.status = 'Aktif'
            GROUP BY 
                a.asset_id, a.asset_type, a.building, a.floor, a.zone, 
                a.instalation_date, a.operational_hours
        """)
        
        results = db.execute(query).mappings().fetchall()
        
        for row in results:
            # Hitung Operating Hours
            operating_hours = 0.0
            if row["instalation_date"] and row["operational_hours"]:
                if isinstance(row["instalation_date"], str):
                    inst_date = datetime.strptime(row["instalation_date"], "%Y-%m-%d").date()
                else:
                    inst_date = row["instalation_date"]
                    
                delta_days = (datetime.now().date() - inst_date).days
                delta_days = max(0, delta_days)
                operating_hours = delta_days * (5.0 / 7.0) * float(row["operational_hours"])
                
            payload = {
                "asset_id": row["asset_id"],
                "Total komplain": row["total_komplain"],
                "Total biaya perbaikan": float(row["total_biaya_perbaikan"]),
                "Lokasi Lantai": row["floor"] or 1,
                "Operating_Hours": round(operating_hours, 2),
                "Tipe": row["asset_type"] or "Unknown",
                "Lokasi Gedung": row["building"] or "Unknown",
                "Lokasi Zona": row["zone"] or "Unknown"
            }
            payloads.append(payload)
            
        return payloads
    finally:
        db.close()

if __name__ == "__main__":
    try:
        print("Mengambil data aset aktif dari database...")
        payloads = get_all_active_payloads()
        
        print(f"Ditemukan {len(payloads)} aset aktif. Memulai pengiriman request ke API...\n")
        
        success_count = 0
        error_count = 0
        
        for payload in payloads:
            asset_id = payload["asset_id"]
            print(f"[*] Memproses Asset ID: {asset_id}...")
            
            try:
                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    resp_data = response.json()
                    print(f"    -> Sukses! RUL: {resp_data.get('predicted_rul'):.2f} tahun")
                    success_count += 1
                else:
                    print(f"    -> Gagal! Status Code: {response.status_code}, Response: {response.text}")
                    error_count += 1
            except Exception as req_e:
                print(f"    -> Error request: {req_e}")
                error_count += 1
                
        print(f"\nSelesai! Berhasil: {success_count}, Gagal: {error_count}")

    except Exception as e:
        print("Error utama:", e)
