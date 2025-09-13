# Multi-stage build for DocuBrain
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/yarn.lock ./
RUN yarn install --frozen-lockfile

COPY frontend/ .
RUN yarn build

# Python backend stage
FROM python:3.11-slim AS backend

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy built frontend
COPY --from=frontend-build /app/frontend/build ./frontend/build

# Install serve for frontend serving
RUN npm install -g serve concurrently

# Environment variables
ENV PORT=8000

# Expose port
EXPOSE $PORT

# Start command
CMD ["sh", "-c", "concurrently \"cd backend && python -m uvicorn server:app --host 0.0.0.0 --port $PORT\" \"serve -s frontend/build -l 3000\""]