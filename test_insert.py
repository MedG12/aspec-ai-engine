import requests
from sqlalchemy import text
from core.database import SessionLocal

def run_insert_test():
    url = "http://localhost:8000/api/insert"
    
    print("Membaca data dari database...")
    db = SessionLocal()
    
    try:
        # Ambil data dari maintenance_logs di-join dengan assets
        # Sesuai dengan skema MaintenanceLogItem yang terbaru
        query = text("""
            SELECT 
                m.ticket_id,
                m.asset_id,
                a.asset_name,
                a.asset_type,
                m.issue_type,
                m.root_cause,
                m.spare_parts_used,
                m.repair_cost
            FROM maintenance_logs m
            JOIN assets a ON m.asset_id = a.asset_id
        """)
        
        results = db.execute(query).mappings().fetchall()
        
        records = []
        for row in results:
            records.append({
                "ticket_id": row["ticket_id"],
                "asset_id": row["asset_id"],
                "asset_name": row["asset_name"] if row["asset_name"] else "",
                "asset_type": row["asset_type"] if row["asset_type"] else "",
                "issue_type": row["issue_type"],
                "root_cause": row["root_cause"],
                "spare_parts_used": row["spare_parts_used"],
                "repair_cost": int(row["repair_cost"]) if row["repair_cost"] is not None else None
            })
            
        if not records:
            print("Tidak ada data di table maintenance_logs.")
            return

        payload = {
            "records": records
        }
        
        print(f"Mengirim {len(records)} baris data ke {url}...")
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            print("Response: SUCCESS")
            print(response.json())
        else:
            print(f"Response: FAILED dengan status {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    run_insert_test()
