#!/bin/bash
for i in {1..100}; do
    curl -X 'POST' \
      'http://localhost:8000/inference' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "request_id": "'"$i"'",
        "model_name": "18e",
        "img_dir": "dataset/example/test1.jpg"
      }' &
done