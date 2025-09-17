FROM python:3.9-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código da aplicação
COPY app.py .

# Expor a porta
EXPOSE 8080

# Comando para iniciar a aplicação
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
