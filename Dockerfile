# Base image for building
ARG LITELLM_BUILD_IMAGE=cgr.dev/chainguard/wolfi-base

# Runtime image
ARG LITELLM_RUNTIME_IMAGE=cgr.dev/chainguard/wolfi-base
# Builder stage
FROM $LITELLM_BUILD_IMAGE AS builder

# Set the working directory to /app
WORKDIR /app

USER root

# Install build dependencies
RUN apk add --no-cache bash gcc py3-pip python3 python3-dev openssl openssl-dev

RUN python -m pip install build

# ============================================
# OPTIMIZATION: Install dependencies FIRST (before copying source code)
# This allows Docker to cache the dependency layer
# ============================================

# Copy only dependency files first
COPY requirements.txt pyproject.toml ./

# Install dependencies as wheels (cached layer)
RUN pip wheel --no-cache-dir --wheel-dir=/wheels/ -r requirements.txt

# ensure pyjwt is used, not jwt
RUN pip uninstall jwt -y || true
RUN pip uninstall PyJWT -y || true
RUN pip install PyJWT==2.9.0 --no-cache-dir

# ============================================
# Now copy the rest of the source code
# Changes to source code will only invalidate layers below this point
# ============================================
COPY . .

# Build Admin UI
RUN chmod +x docker/build_admin_ui.sh && ./docker/build_admin_ui.sh

# Build the package
RUN rm -rf dist/* && python -m build

# There should be only one wheel file now, assume the build only creates one
RUN ls -1 dist/*.whl | head -1

# Install the package
RUN pip install dist/*.whl

# Runtime stage
FROM $LITELLM_RUNTIME_IMAGE AS runtime

# Ensure runtime stage runs as root
USER root

# Install runtime dependencies
RUN apk add --no-cache bash openssl tzdata nodejs npm python3 py3-pip

WORKDIR /app

# ============================================
# OPTIMIZATION: Copy and install pre-built wheels FIRST
# This layer is cached as long as requirements.txt doesn't change
# ============================================
COPY --from=builder /wheels/ /wheels/

# Install wheels (cached layer)
RUN pip install /wheels/* --no-index --find-links=/wheels/ && rm -rf /wheels

# ============================================
# Now copy the application code and built wheel
# ============================================
COPY --from=builder /app/dist/*.whl .
COPY . .

RUN ls -la /app

# Install the built application wheel
RUN pip install *.whl --no-deps && rm -f *.whl

# Remove test files and keys from dependencies
RUN find /usr/lib -type f -path "*/tornado/test/*" -delete && \
    find /usr/lib -type d -path "*/tornado/test" -delete

# Install semantic_router and aurelio-sdk using script
RUN chmod +x docker/install_auto_router.sh && ./docker/install_auto_router.sh

# Generate prisma client
RUN prisma generate
RUN chmod +x docker/entrypoint.sh
RUN chmod +x docker/prod_entrypoint.sh

EXPOSE 4000/tcp

RUN apk add --no-cache supervisor
COPY docker/supervisord.conf /etc/supervisord.conf

ENTRYPOINT ["docker/prod_entrypoint.sh"]

# Append "--detailed_debug" to the end of CMD to view detailed debug logs
CMD ["--port", "4000"]
