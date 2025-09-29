FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt 

COPY api.py .

COPY model/ model/

EXPOSE 9000

CMD ["python3", "api.py"]