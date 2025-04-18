#!/bin/bash

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral",
    "messages": [{"role": "user", "content": "Je suis un nouveau papa. Comment soutenir ma femme ?"}]
}'
