"""法规管理模块 - 加载和管理审查要点与法规要求的映射"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

# 法规配置文件路径
_REGULATIONS_FILE = Path(__file__).parent / "regulations.yaml"

# 内存中的法规缓存（应用启动时加载一次）
_regulations_cache: Dict[str, str] | None = None


def load_regulations(reload: bool = False) -> Dict[str, str]:
    """
    加载法规配置文件到内存。
    
    Args:
        reload: 是否强制重新加载（默认使用缓存）
        
    Returns:
        审查要点 -> 法规要求的字典
        
    Raises:
        FileNotFoundError: 如果配置文件不存在
        yaml.YAMLError: 如果配置文件格式错误
    """
    global _regulations_cache
    
    if _regulations_cache is not None and not reload:
        return _regulations_cache
    
    if not _REGULATIONS_FILE.exists():
        raise FileNotFoundError(
            f"法规配置文件不存在: {_REGULATIONS_FILE}\n"
            "请创建 regulations.yaml 文件并配置审查要点与法规要求的映射。"
        )
    
    with open(_REGULATIONS_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    if not isinstance(data, dict):
        raise ValueError(
            f"法规配置文件格式错误: 期望字典格式，实际为 {type(data).__name__}"
        )
    
    # 验证并清理空的法规数据
    _regulations_cache = {
        str(key).strip(): str(value).strip()
        for key, value in data.items()
        if value is not None  # 跳过注释掉的条目
    }
    
    return _regulations_cache


def get_regulation(check_point: str) -> str:
    """
    根据审查要点获取对应的法规要求。
    
    Args:
        check_point: 审查要点名称（如"主体资格审查"）
        
    Returns:
        对应的法规要求文本
        
    Raises:
        KeyError: 如果审查要点不存在
    """
    regulations = load_regulations()
    check_point = check_point.strip()
    
    if check_point not in regulations:
        available = list(regulations.keys())
        raise KeyError(
            f"未找到审查要点: '{check_point}'\n"
            f"可用的审查要点: {available}"
        )
    
    return regulations[check_point]


def list_check_points() -> List[str]:
    """
    获取所有可用的审查要点列表。
    
    Returns:
        审查要点名称列表
    """
    regulations = load_regulations()
    return list(regulations.keys())


def get_all_regulations() -> Dict[str, str]:
    """
    获取所有法规配置。
    
    Returns:
        审查要点 -> 法规要求的完整字典
    """
    return load_regulations().copy()
