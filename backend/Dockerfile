# SnapNotes backend image.
#
#   docker build -t snapnotes .
#   docker run --rm -p 8000:8000 -v snapnotes-data:/data \
#       -e SNAPNOTES_TOKEN=<long random string> -e ANTHROPIC_API_KEY=<key> snapnotes
#
# Run exactly one container: the job worker is in-process and the database is a single SQLite file on /data.
#
# Dependencies come from requirements.lock (exact versions, every file hash-checked), so a rebuild next month installs
# the same bytes that were tested. Regenerate it with the command at the top of that file (see DEPLOY.md).
#
# The same suite that runs on a laptop can run in this exact environment, without starting anything:
#   docker build --target test .
# (Needs BuildKit, the default since Docker 23; a plain `docker build .` skips the test stage.)

# The lock files are resolved for this Python version and the whole test suite has run on it: change both together.
ARG PYTHON_VERSION=3.14

# --- build stage: install the locked dependencies, then the app, into a virtualenv ------------------------------
FROM python:${PYTHON_VERSION}-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
RUN python -m venv /opt/venv
# Dependencies first and on their own: this layer is rebuilt only when the lock file changes.
COPY requirements.lock ./
RUN /opt/venv/bin/pip install --require-hashes --no-deps -r requirements.lock
COPY pyproject.toml ./
COPY app ./app
# --no-deps: everything is already installed and verified above, so pip must not go looking for more.
RUN /opt/venv/bin/pip install --no-deps .

# --- test stage (opt-in: docker build --target test .) -----------------------------------------------------------
FROM build AS test
COPY requirements-dev.lock ./
RUN /opt/venv/bin/pip install --require-hashes --no-deps -r requirements-dev.lock
COPY tests ./tests
RUN /opt/venv/bin/python -m pytest -q

# --- runtime stage ---------------------------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SNAPNOTES_DATA_DIR=/data \
    SNAPNOTES_DOCS=0
COPY --from=build /opt/venv /opt/venv
RUN useradd --system --uid 10001 --home-dir /data --shell /usr/sbin/nologin snapnotes \
    && mkdir -p /data && chown snapnotes:snapnotes /data
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# The entrypoint starts as root only to fix ownership of a freshly mounted volume, then drops to the app user.
ENTRYPOINT ["docker-entrypoint.sh"]
# One process on purpose (see above). --proxy-headers lets the app see the real client address behind Fly's proxy.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
