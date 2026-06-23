# Imagen base ligera de Python
FROM python:3.10-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias primero (mejor aprovechamiento de la cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Puerto que expone Dash / Gunicorn
EXPOSE 8050

# Servir la aplicación con Gunicorn (app:server)
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "app:server"]
