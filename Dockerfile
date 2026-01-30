FROM python:3.11-slim

# Install system dependencies for cartopy
RUN apt-get update && apt-get install -y \
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY newsletter-builder/requirements.txt ./newsletter-builder/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r newsletter-builder/requirements.txt

# Copy application code
COPY . .

# Set working directory to the app
WORKDIR /app/newsletter-builder

# Expose port
EXPOSE 10000

# Run with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
