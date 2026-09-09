# SIH26124 - Backend Observation Foundation & Geospatial Event Fusion (Step 2)

This is the FastAPI backend for ingesting geotagged AI detections and fusing them into clustered Geospatial Events.

## Architecture

1. **Ingestion**: A bus submits an `Observation`.
2. **Geospatial Fusion**: The Haversine distance is calculated between the new observation and all active events of the same class type.
3. **Clustering**: If the observation is within 10 meters of an existing event, it is fused (counts updated, location averaged). Otherwise, a new event is created.
4. **Concurrency**: Row-level locking (`FOR UPDATE`) is used to ensure transaction safety during the fusion process.

## Setup Instructions

1. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Database:**
   - Create a MySQL database (default expected name is `sih26124`).
   - Edit the `.env` file with your MySQL credentials.

4. **Run the Server:**
   ```bash
   uvicorn app:app --reload
   ```

5. **Run the Tests:**
   The test suite runs on an isolated in-memory SQLite database, so it won't affect your MySQL data.
   ```bash
   pytest test_api.py test_fusion.py -v
   ```

6. **Run the Fleet Simulator:**
   Simulate multiple buses sending concurrent observations to test the fusion logic in real-time.
   ```bash
   python fleet_simulator.py
   ```

## API Documentation
Once the server is running, visit:
- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
