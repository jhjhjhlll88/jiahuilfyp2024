#!/bin/bash

# Ensure you're using python3 to avoid version issues
python3 -m pip install -r requirements.txt

# Run Django's collectstatic command
python3 manage.py collectstatic --noinput
