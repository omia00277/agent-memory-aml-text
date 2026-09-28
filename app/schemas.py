from pydantic import BaseModel, Field
from typing import List, Optional, Union
from datetime import datetime



# BaseModel是数据模型类，可用于  定义消息结构 + 自动校验 + 类型转换
# Field是函数，用来给模型字段添加额外配置（约束、默认值、描述等）

# Union：多选一
# Optional：可以是某类型，也可以是 None

class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: Union[str, List[dict]]
    timestamp: Optional[int] = None


class AddRequest(BaseModel):
    request_id: str
    messages: List[Message]
    user_id: str
    session_id: str


class AddResponse(BaseModel):
    success: bool = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: Union[str, List[dict]]
    options: Optional[List[str]] = None
    user_id: str
    top_k: int = 100


class MemoryItem(BaseModel):
    id: str
    content: Union[str, List[dict]]
    score: Optional[float] = None
    created_at: Optional[str] = None


class SearchResponse(BaseModel):
    data: List[MemoryItem]
