#!/usr/bin/env bash

# set up all needed env variables
export APP_SETTINGS="app.config.DevelopmentConfig"
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/apartments_dev"

python app/scrapy_client.py
