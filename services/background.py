from apscheduler.schedulers.background import BackgroundScheduler
import logging

logger = logging.getLogger(__name__)

# Initialize APScheduler
scheduler = BackgroundScheduler()

from core.database import SessionLocal
from services.clustering_service import run_clustering_job
from services.recommendation_service import run_narrative_recommendation_job
from services.retrain_service import retrain_xgboost_model

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

def retrain_xgboost_weekly():
    """
    Retrain XGBoost RUL model every week using fresh data from the database.
    """
    logger.info("Running scheduled task: retrain_xgboost_weekly")
    try:
        result = retrain_xgboost_model()
        logger.info(f"XGBoost retrain result: {result}")
    except Exception as e:
        logger.error(f"Error during scheduled XGBoost retrain job: {e}")
    logger.info("XGBoost retrain task completed.")

def start_scheduler():
    # Schedule the job to run every 12 hours (misal jam 00:00 dan 12:00)
    scheduler.add_job(update_clusters_and_rul, 'cron', hour=12, minute=0)
    # scheduler.add_job(update_clusters_and_rul, 'interval', minutes=1)
    
    # Schedule the narrative recommendation job to run every 1 hour
    scheduler.add_job(update_narrative_recommendations, 'interval', hours=1)
    
    # Schedule XGBoost retrain every Sunday at 02:00
    scheduler.add_job(retrain_xgboost_weekly, 'cron', day_of_week='sun', hour=2, minute=0)
    
    scheduler.start()
    logger.info("APScheduler started (Clustering: every 12h, Narrative: every 1h, XGBoost Retrain: weekly Sunday 02:00).")

def stop_scheduler():
    scheduler.shutdown()
    logger.info("APScheduler shut down.")
