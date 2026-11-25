from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["target"] == "wait_min"


def test_valid_prediction():
    response = client.post(
        "/predict",
        json={
            "patients_ahead": 10,
            "doctors_on_duty": 3,
            "arrival_hour": 11,
            "avg_consult_min": 8.5,
            "lab_tests_pending": 2,
            "distance_km": 5.0,
            "department": "Cardiology",
            "wing": "New Wing"
        }
    )

    assert response.status_code == 200
    assert response.json()["prediction"] > 0


def test_negative_patients_rejected():
    response = client.post(
        "/predict",
        json={
            "patients_ahead": -1,
            "doctors_on_duty": 3,
            "arrival_hour": 11,
            "avg_consult_min": 8.5,
            "lab_tests_pending": 2,
            "distance_km": 5.0,
            "department": "Cardiology",
            "wing": "New Wing"
        }
    )

    assert response.status_code == 422


def test_distance_above_70_rejected():
    response = client.post(
        "/predict",
        json={
            "patients_ahead": 10,
            "doctors_on_duty": 3,
            "arrival_hour": 11,
            "avg_consult_min": 8.5,
            "lab_tests_pending": 2,
            "distance_km": 75,
            "department": "Cardiology",
            "wing": "New Wing"
        }
    )

    assert response.status_code == 422