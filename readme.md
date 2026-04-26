# IoT ML IDS

Machine learning-based intrusion detection system for IoT traffic.

## Run with Docker
docker build -t iot-ids-api .
docker run -p 8000:8000 iot-ids-api

Open:
http://localhost:8000/docs

## API
POST /predict

Example:
{
  "features": {
    "Protocol Type": 6,
    "Flow Duration": 120,
    "Tot Fwd Pkts": 5,
    "Tot Bwd Pkts": 3
  }
}

Response:
{
  "prediction": "DNS_SPOOFING",
  "family": "SPOOF_SCAN",
  "stage1": "ATTACK"
}
