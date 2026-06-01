from apscheduler.schedulers.background import BackgroundScheduler
import logging

logger = logging.getLogger(__name__)

# Initialize APScheduler
scheduler = BackgroundScheduler()

from core.database import SessionLocal
from services.clustering_service import run_clustering_job
from services.recommendation_service import run_narrative_recommendation_job

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

def update_narrative_recommendations():
    """
    Generate narrative recommendations for assets every 1 hour.
    """
    logger.info("Running scheduled task: update_narrative_recommendations (RAG narrative LLM)")
    db = SessionLocal()
    try:
        run_narrative_recommendation_job(db)
    except Exception as e:
        logger.error(f"Error during scheduled narrative recommendation job: {e}")
    finally:
        db.close()
    logger.info("Narrative recommendation task completed.")

def start_scheduler():
    # Schedule the job to run every 12 hours (misal jam 00:00 dan 12:00)
    scheduler.add_job(update_clusters_and_rul, 'cron', hour=12, minute=0)
    
    # Schedule the narrative recommendation job to run every 1 hour
    scheduler.add_job(update_narrative_recommendations, 'interval', hours=1)
    
    scheduler.start()
    logger.info("APScheduler started (Clustering job set for every 12 hours, Narrative job set for every 1 hour).")

def stop_scheduler():
    scheduler.shutdown()
    logger.info("APScheduler shut down.")
