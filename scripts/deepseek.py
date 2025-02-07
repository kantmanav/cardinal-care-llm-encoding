#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 20:51:03 2025

@author: sareh
"""

import os
import requests

def run_deepseek_r1(prompt, model="deepseek-reasoner"):
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise ValueError("DeepSeek API key not found. Please set the environment variable 'DEEPSEEK_API_KEY'.")

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {deepseek_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}  # Removed 'system' message
        ],
        "temperature": 1.0,
        "top_p": 1.0
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raises an error if request fails
        
        data = response.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0].get("message", {}).get("content", "No response content.")
        else:
            raise ValueError("Unexpected response structure from DeepSeek API.")

    except requests.exceptions.RequestException as e:
        print(f"Error calling DeepSeek API: {e}")
        return None

# Test the function
if __name__ == "__main__":
    prompt_text = "What is the capital of France?"
    response = run_deepseek_r1(prompt_text)
    print("DeepSeek Response:", response)