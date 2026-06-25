FROM python:3.12-slim

WORKDIR /app

COPY infra/requirements.txt .

COPY scripts/ /app/scripts/
COPY configs/ /app/configs/

RUN pip install -r requirements.txt

CMD ["python", "scripts/main.py"]
