#!/bin/bash

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral",
    "messages": [
      {
        "role": "user",
        "content": "My partner is pregnant and can’t sleep well. How can I support her?"
      }
    ]
  }' | jq