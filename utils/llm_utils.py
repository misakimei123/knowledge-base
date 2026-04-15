"""
LLM 调用工具 - Ollama 封装
"""
import logging
from typing import AsyncGenerator, Dict, Any, Optional

logger = logging.getLogger(__name__)


async def call_ollama(
    prompt: str,
    model: str = "qwen2.5-14b-chat",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    system_prompt: Optional[str] = None
) -> str:
    """
    调用 Ollama LLM（非流式）
    :param prompt: 用户提示
    :param model: 模型名称
    :param base_url: Ollama 服务地址
    :param temperature: 温度参数
    :param max_tokens: 最大生成长度
    :param system_prompt: 系统提示
    :return: 生成的文本
    """
    try:
        import aiohttp
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base_url}/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "")
                else:
                    logger.error(f"Ollama API error: {response.status}")
                    return ""
    
    except Exception as e:
        logger.error(f"Failed to call Ollama: {e}", exc_info=True)
        return ""


async def stream_ollama(
    prompt: str,
    model: str = "qwen2.5-14b-chat",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    system_prompt: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    调用 Ollama LLM（流式输出）
    :param prompt: 用户提示
    :param model: 模型名称
    :param base_url: Ollama 服务地址
    :param temperature: 温度参数
    :param max_tokens: 最大生成长度
    :param system_prompt: 系统提示
    :yield: 生成的文本片段
    """
    try:
        import aiohttp
        import json
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base_url}/api/generate", json=payload) as response:
                if response.status == 200:
                    async for line in response.content.iter_any():
                        # 解析 NDJSON 格式
                        for chunk in line.split(b'\n'):
                            if chunk.strip():
                                try:
                                    data = json.loads(chunk)
                                    if "response" in data:
                                        yield data["response"]
                                except json.JSONDecodeError:
                                    continue
                else:
                    logger.error(f"Ollama API error: {response.status}")
    
    except Exception as e:
        logger.error(f"Failed to stream Ollama: {e}", exc_info=True)


async def check_ollama_health(base_url: str = "http://localhost:11434") -> bool:
    """检查 Ollama 服务健康状态"""
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/api/tags") as response:
                return response.status == 200
    
    except Exception:
        return False


async def list_ollama_models(base_url: str = "http://localhost:11434") -> list:
    """列出 Ollama 可用模型"""
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/api/tags") as response:
                if response.status == 200:
                    result = await response.json()
                    return [model["name"] for model in result.get("models", [])]
    
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return []


# RAG 生成专用函数
async def generate_rag_answer(
    query: str,
    contexts: list,
    model: str = "qwen2.5-14b-chat",
    base_url: str = "http://localhost:11434"
) -> Dict[str, Any]:
    """
    基于检索上下文生成答案
    :param query: 用户查询
    :param contexts: 检索到的上下文
    :param model: 模型名称
    :param base_url: Ollama 服务地址
    :return: 包含答案和元数据的字典
    """
    # 构建提示词
    context_text = "\n\n".join([f"[参考{i+1}]: {ctx}" for i, ctx in enumerate(contexts[:5])])
    
    system_prompt = """你是一个专业的企业知识库助手。请根据提供的参考信息回答问题。
要求：
1. 严格基于参考信息回答，不要编造
2. 如果参考信息不足，请说明
3. 回答要简洁清晰
4. 如有必要，可以引用参考来源"""

    prompt = f"""参考信息：
{context_text}

用户问题：{query}

请根据以上参考信息回答问题："""

    answer = await call_ollama(
        prompt=prompt,
        model=model,
        base_url=base_url,
        temperature=0.5,  # 较低温度提高准确性
        system_prompt=system_prompt
    )
    
    return {
        "answer": answer,
        "contexts_used": len(contexts),
        "model": model
    }
