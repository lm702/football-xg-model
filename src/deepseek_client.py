import requests
import json
import streamlit as st

def get_api_key():
    """从 Streamlit secrets 获取 API Key"""
    try:
        return st.secrets["DEEPSEEK_API_KEY"]
    except KeyError:
        st.error("未配置 DeepSeek API Key。请在 .streamlit/secrets.toml 中设置 DEEPSEEK_API_KEY")
        return None

def deepseek_chat(prompt, model="deepseek-chat", temperature=0.6, max_tokens=800):
    """
    调用 DeepSeek API (兼容 OpenAI 格式) 进行对话。
    """
    api_key = get_api_key()
    if not api_key:
        return None

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        st.error("AI 请求超时，请稍后重试")
    except requests.exceptions.RequestException as e:
        st.error(f"AI 请求失败: {e}")
    except (KeyError, IndexError):
        st.error("AI 返回格式异常")
    return None
