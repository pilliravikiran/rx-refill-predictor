# A Dockerfile is a recipe. Each line is one step, and Docker caches each one.

# STEP 1 - what we start from.
# "slim" is a stripped-down Linux with Python already installed. About 130MB.
# The full image is nearly 1GB and we do not need a C compiler in production.
FROM python:3.11-slim

# STEP 2 - the folder inside the box where everything lives.
WORKDIR /code

# STEP 3 - dependencies FIRST, on their own.
# This is the one trick worth understanding. Docker caches every step, and
# reuses the cache until something changes. Your library list changes maybe
# once a month. Your code changes ten times an hour.
#
# By copying requirements on its own line, a code change reuses the cached
# install and rebuilds in seconds instead of minutes.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# STEP 4 - now the things that change often.
COPY app.py .
COPY models/refill_model.pkl   models/refill_model.pkl
COPY models/forecast_model.pkl models/forecast_model.pkl

# STEP 5 - tell anyone reading this which port the app listens on.
EXPOSE 8000

# STEP 6 - the command that runs when the container starts.
# --host 0.0.0.0 is REQUIRED. The default is 127.0.0.1, which inside a
# container means "only reachable from inside this container" - so your
# browser gets nothing and the container looks broken.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
