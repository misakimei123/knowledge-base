"""
文件处理工具函数
"""
import hashlib
import os
import logging
from typing import Generator, Optional

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """
    计算文件哈希值（用于去重）
    :param file_path: 文件路径
    :param algorithm: 哈希算法
    :return: 文件哈希值
    """
    hash_func = getattr(hashlib, algorithm, hashlib.sha256)
    hasher = hash_func()
    
    try:
        with open(file_path, 'rb') as f:
            # 分块读取，避免大文件内存溢出
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    except Exception as e:
        logger.error(f"Failed to compute hash for {file_path}: {e}")
        return ""


def read_file_chunks(file_path: str, chunk_size: int = 8192) -> Generator[bytes, None, None]:
    """
    分块读取文件
    :param file_path: 文件路径
    :param chunk_size: 块大小
    :yield: 文件块
    """
    try:
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")


def get_file_info(file_path: str) -> dict:
    """获取文件信息"""
    try:
        stat = os.stat(file_path)
        return {
            "file_name": os.path.basename(file_path),
            "file_size": stat.st_size,
            "file_ext": os.path.splitext(file_path)[1].lower(),
            "modified_time": stat.st_mtime
        }
    except Exception as e:
        logger.error(f"Failed to get file info: {e}")
        return {}


def is_supported_file(file_path: str) -> bool:
    """检查是否为支持的文件类型"""
    supported_extensions = {
        '.pdf', '.docx', '.doc', '.txt', '.md',
        '.xlsx', '.xls', '.pptx', '.ppt',
        '.csv', '.json', '.xml', '.html'
    }
    
    ext = os.path.splitext(file_path)[1].lower()
    return ext in supported_extensions


def ensure_directory(dir_path: str):
    """确保目录存在"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")


def safe_filename(filename: str) -> str:
    """生成安全的文件名"""
    # 移除非法字符
    invalid_chars = '<>:"/\\|？*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename.strip()
