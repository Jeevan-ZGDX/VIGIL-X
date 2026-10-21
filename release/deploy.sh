#!/usr/bin/env bash
# ==============================================================================
# VIGIL-X Production Deployment & Release Installer
# Multimodal Pedestrian & Path Anomaly Safety System (ISO 3691-4 / ISO 13849)
# ==============================================================================

set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BOLD}======================================================${NC}"
echo -e "${BOLD}  VIGIL-X Edge Safety System: Deploying Release v1.0.0 ${NC}"
echo -e "${BOLD}======================================================${NC}"

# 1. Check prerequisites
echo -e "\n${YELLOW}[1/5] Checking environment dependencies...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed. Please install Docker before proceeding.${NC}"
    exit 1
fi

COMPOSE_CMD=""
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    echo -e "${RED}Error: docker compose is not installed.${NC}"
    exit 1
fi
echo -e "${GREEN}Docker and ${COMPOSE_CMD} are available.${NC}"

# 2. Check for offline image tarball or local image
echo -e "\n${YELLOW}[2/5] Checking Docker image assets...${NC}"
IMAGE_TAG="vigil-x:v1.0.0"
TAR_ARCHIVE="vigil-x-v1.0.0-image.tar.gz"

if docker image inspect "${IMAGE_TAG}" &> /dev/null; then
    echo -e "${GREEN}Found local Docker image: ${IMAGE_TAG}${NC}"
elif [ -f "${TAR_ARCHIVE}" ]; then
    echo -e "${YELLOW}Loading image from offline archive ${TAR_ARCHIVE}...${NC}"
    gzip -dc "${TAR_ARCHIVE}" | docker load
    echo -e "${GREEN}Successfully loaded ${IMAGE_TAG}${NC}"
else
    echo -e "${YELLOW}Image not found locally. Building from source...${NC}"
    docker build -t "${IMAGE_TAG}" -t vigil-x:latest ..
fi

# 3. Detect Video Capture Devices
echo -e "\n${YELLOW}[3/5] Detecting camera hardware...${NC}"
if [ -e "/dev/video0" ]; then
    echo -e "${GREEN}Found physical camera on /dev/video0.${NC}"
else
    echo -e "${YELLOW}Notice: No physical camera on /dev/video0. System will run in synthetic/replay mode.${NC}"
fi

# 4. Prepare local volume mounts
echo -e "\n${YELLOW}[4/5] Initializing persistent data directories...${NC}"
mkdir -p data config
if [ ! -f "data/events.jsonl" ]; then
    touch data/events.jsonl
fi
if [ ! -f "config/default_config.yaml" ] && [ -f "../config/default_config.yaml" ]; then
    cp -r ../config ./
fi

# 5. Launch containers
echo -e "\n${YELLOW}[5/5] Starting VIGIL-X Safety System via ${COMPOSE_CMD}...${NC}"
${COMPOSE_CMD} down --remove-orphans 2>/dev/null || true
${COMPOSE_CMD} up -d

echo -e "\n${GREEN}Waiting for system healthcheck...${NC}"
for i in {1..15}; do
    if curl -s http://localhost:8000/health | grep -q '"status":"UP"'; then
        echo -e "${GREEN}VIGIL-X Safety System is HEALTHY and OPERATIONAL!${NC}"
        echo -e "${BOLD}Dashboard URL: ${GREEN}http://localhost:8000${NC}"
        echo -e "${BOLD}Health Endpoint: ${GREEN}http://localhost:8000/health${NC}"
        echo -e "${BOLD}REST API Status: ${GREEN}http://localhost:8000/api/status${NC}"
        exit 0
    fi
    sleep 2
done

echo -e "${YELLOW}Warning: Container started but health check took longer than expected.${NC}"
echo -e "Check logs with: ${BOLD}${COMPOSE_CMD} logs -f${NC}"
