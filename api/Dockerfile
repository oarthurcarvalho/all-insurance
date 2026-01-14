FROM python:3.12-slim

# o LightGBM depende da biblioteca OpenMP (libgomp), que não vem na imagem slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# o Heroku informa a porta pela variável de ambiente PORT
CMD exec gunicorn handler:app --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:${PORT:-10000}
