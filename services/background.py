from apscheduler.schedulers.background import BackgroundScheduler
import logging

logger = logging.getLogger(__name__)

# Initialize APScheduler
scheduler = BackgroundScheduler()

from core.database import SessionLocal
from services.clustering_service import run_clustering_job

def update_clusters_and_rul():
    """
    Batch clustering and updating DB every 12 hours.
    """
    logger.info("Running scheduled task: update_clusters_and_rul (Auto Clustering)")
    db = SessionLocal()
    try:
        run_clustering_job(db)
        # RUL updates can be added here later
    except Exception as e:
        logger.error(f"Error during scheduled clustering job: {e}")
    finally:
        db.close()
    logger.info("Scheduled task completed successfully.")

def start_scheduler():
    # Schedule the job to run every 12 hours (misal jam 00:00 dan 12:00)
    scheduler.add_job(update_clusters_and_rul, 'cron', hour='0,12', minute=0)
    scheduler.start()
    logger.info("APScheduler started (Clustering job set for every 12 hours).")

def stop_scheduler():
    scheduler.shutdown()
    logger.info("APScheduler shut down.")
