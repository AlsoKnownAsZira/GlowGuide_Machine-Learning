# Gunakan base image Python versi slim-bookworm
FROM python:3.11.10-slim-bookworm

# Set working directory di dalam container
WORKDIR /app

# Salin file requirements.txt ke dalam container
COPY requirements.txt .

# Tambahkan repositori yang valid, perbaiki paket, dan instal dependensi
RUN echo "deb http://deb.debian.org/debian bookworm main" > /etc/apt/sources.list \
    && echo "deb http://deb.debian.org/debian bookworm-updates main" >> /etc/apt/sources.list \
    && echo "deb http://deb.debian.org/debian-security bookworm-security main" >> /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Instal dependensi Python dari requirements.txt tanpa menyimpan cache
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file ke dalam container
COPY . .

# Buka port untuk Flask API
EXPOSE 5000

# Jalankan aplikasi menggunakan Python
CMD ["flask", "run", "--host=0.0.0.0"]
