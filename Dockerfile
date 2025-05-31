FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# Copy application code
COPY . .

# Set environment variable
ENV GROQ_API_KEY=${GROQ_API_KEY}

# Command to run the application
CMD ["sh", "start.sh"]