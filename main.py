"""
企业级知识库 - 主入口
FastAPI + Gradio 整合应用
"""
import asyncio
import logging
import time
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 导入核心模块
from config.settings import Settings, get_settings, load_config
from core.rag_engine import HybridRAGEngine
from core.kg_engine import Neo4jKGEvaluator
from core.pipeline import DocumentPipeline, ProcessingResult
from core.confidence.evaluator import ConfidenceEvaluator
from core.confidence.types import RequestContext
from storage.milvus_client import MilvusClient
from storage.neo4j_client import Neo4jClient
from storage.metadata_db import MetadataDB
from utils.llm_utils import generate_rag_answer, check_ollama_health

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# 全局状态
class AppState:
    """应用状态管理"""
    def __init__(self):
        self.settings = None
        self.milvus = None
        self.neo4j = None
        self.metadata_db = None
        self.rag_engine = None
        self.pipeline = None
        self.confidence_evaluator = None
        self.initialized = False


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("Initializing application...")
    
    settings = get_settings()
    state.settings = settings
    
    # 初始化存储
    state.milvus = MilvusClient(uri=settings.milvus_uri)
    state.neo4j = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password
    )
    state.metadata_db = MetadataDB()
    state.metadata_db.connect()
    
    # 初始化引擎
    state.rag_engine = HybridRAGEngine(
        milvus_client=state.milvus,
        neo4j_client=state.neo4j,
        config={
            "vector_top_k": settings.vector_top_k,
            "kg_top_k": settings.kg_top_k,
            "final_top_k": settings.final_top_k,
            "rerank_enable": settings.rerank_enable
        }
    )
    
    state.pipeline = DocumentPipeline(config={
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap
    })
    
    state.confidence_evaluator = ConfidenceEvaluator()
    
    state.initialized = True
    logger.info("Application initialized successfully")
    
    yield
    
    # 关闭时清理
    logger.info("Shutting down application...")
    if state.metadata_db:
        state.metadata_db.close()
    if state.neo4j:
        await state.neo4j.close()
    if state.milvus:
        await state.milvus.close()


# 创建 FastAPI 应用
app = FastAPI(
    title="Enterprise Knowledge Base",
    description="RAG+Knowledge Graph 混合检索的企业级知识库系统",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ API Models ============

class QueryRequest(BaseModel):
    """查询请求"""
    query: str
    category: Optional[str] = None
    user_id: str = "default"
    top_k: int = 8


class QueryResponse(BaseModel):
    """查询响应"""
    answer: str
    confidence: float
    confidence_level: str
    explanation: str
    sources: List[Dict[str, Any]]
    audit_id: str
    response_time_ms: int


class UploadResponse(BaseModel):
    """上传响应"""
    doc_id: str
    file_name: str
    category: str
    status: str
    vector_count: int
    entity_count: int


# ============ API Routes ============

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy" if state.initialized else "initializing",
        "milvus": "connected" if state.milvus else "disconnected",
        "neo4j": "connected" if state.neo4j else "disconnected",
        "metadata_db": "connected" if state.metadata_db else "disconnected"
    }


@app.post("/api/query", response_model=QueryResponse)
async def query_knowledge(request: QueryRequest):
    """
    知识检索接口
    """
    start_time = time.time()
    
    if not state.initialized:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # 1. 执行混合检索
        category_filter = [request.category] if request.category else None
        
        hybrid_result = await state.rag_engine.retrieve(
            query=request.query,
            category_filter=category_filter
        )
        
        # 2. 从结果中提取上下文
        contexts = [
            item.get("chunk_text", item.get("text", ""))
            for item in hybrid_result.merged_results[:5]
        ]
        
        # 3. 生成答案
        rag_result = await generate_rag_answer(
            query=request.query,
            contexts=contexts,
            model=state.settings.llm_model
        )
        
        # 4. 置信度评估
        request_context = RequestContext(
            user_id=request.user_id,
            category=request.category or "",
            query_type="general"
        )
        
        confidence_result = await state.confidence_evaluator.evaluate(
            query=request.query,
            vector_results=hybrid_result.vector_results,
            kg_results=hybrid_result.kg_results,
            answer=rag_result["answer"],
            contexts=contexts,
            request_context=request_context
        )
        
        # 5. 记录日志
        response_time = int((time.time() - start_time) * 1000)
        
        if state.metadata_db:
            state.metadata_db.log_query(
                user_id=request.user_id,
                query_text=request.query,
                category_filter=str(category_filter),
                confidence_score=confidence_result.confidence,
                confidence_level=confidence_result.level,
                response_time_ms=response_time
            )
            
            # 保存置信度记录
            state.metadata_db.save_confidence_record({
                "audit_id": confidence_result.audit_id,
                "confidence_score": confidence_result.confidence,
                "confidence_level": confidence_result.level,
                "vector_signal": confidence_result.signals.get("vector", {}).score if confidence_result.signals else 0,
                "kg_signal": confidence_result.signals.get("kg", {}).score if confidence_result.signals else 0,
                "faithfulness_signal": confidence_result.signals.get("faithfulness", {}).score if confidence_result.signals else 0,
                "explanation": confidence_result.explanation
            })
        
        # 6. 构建响应
        sources = []
        for item in hybrid_result.merged_results[:request.top_k]:
            sources.append({
                "id": item.get("id"),
                "type": "vector" if "vector" in item.get("sources", []) else "kg",
                "score": item.get("fusion_score", 0),
                "content": item.get("chunk_text", item.get("text", ""))[:500]
            })
        
        return QueryResponse(
            answer=rag_result["answer"],
            confidence=confidence_result.confidence,
            confidence_level=confidence_result.level,
            explanation=confidence_result.explanation,
            sources=sources,
            audit_id=confidence_result.audit_id,
            response_time_ms=response_time
        )
    
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form("general"),
    user_id: str = Form("default")
):
    """
    文档上传接口
    """
    if not state.initialized:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # 保存文件（临时）
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # 生成文档 ID
        import hashlib
        doc_id = hashlib.md5(f"{file.filename}_{time.time()}".encode()).hexdigest()[:16]
        
        # 处理文档
        result = await state.pipeline.process(
            doc_id=doc_id,
            file_path=tmp_path,
            category=category
        )
        
        # 更新元数据
        if state.metadata_db and result.success:
            state.metadata_db.insert_document({
                "doc_id": doc_id,
                "file_name": file.filename,
                "file_path": tmp_path,
                "category": category,
                "status": "processed"
            })
            
            state.metadata_db.update_document_status(
                doc_id=doc_id,
                status="processed",
                vector_count=result.vector_count,
                entity_count=result.entity_count
            )
        
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        return UploadResponse(
            doc_id=doc_id,
            file_name=file.filename,
            category=category,
            status="processed",
            vector_count=result.vector_count,
            entity_count=result.entity_count
        )
    
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
async def list_documents(
    category: Optional[str] = None,
    limit: int = 50
):
    """列出文档"""
    if not state.metadata_db:
        return []
    
    return state.metadata_db.list_documents(category=category, limit=limit)


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """删除文档"""
    if not state.metadata_db:
        raise HTTPException(status_code=503, detail="Metadata DB not available")
    
    state.metadata_db.delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}


@app.get("/api/categories")
async def get_categories():
    """获取分类列表"""
    if not state.metadata_db:
        return []
    
    return state.metadata_db.get_categories()


# ============ Gradio 界面 ============

def create_gradio_interface():
    """创建 Gradio 界面"""
    try:
        import gradio as gr
        
        async def search(query: str, category: str, user_id: str):
            """Gradio 搜索回调"""
            try:
                request = QueryRequest(
                    query=query,
                    category=category if category != "全部" else None,
                    user_id=user_id or "default"
                )
                
                response = await query_knowledge(request)
                
                result = f"""### 回答
{response.answer}

### 置信度评估
- **置信分数**: {response.confidence:.2%}
- **等级**: {response.confidence_level}
- **解释**: {response.explanation}

### 参考来源
"""
                for i, source in enumerate(response.sources, 1):
                    result += f"\n{i}. [{source['type']}] (相似度：{source['score']:.2f})\n   {source['content'][:200]}..."
                
                return result
            
            except Exception as e:
                return f"Error: {str(e)}"
        
        with gr.Blocks(title="企业知识库") as demo:
            gr.Markdown("# 📚 企业知识库检索系统")
            
            with gr.Row():
                with gr.Column(scale=3):
                    query_input = gr.Textbox(
                        label="问题",
                        placeholder="请输入您的问题...",
                        lines=2
                    )
                    
                    with gr.Row():
                        category_dropdown = gr.Dropdown(
                            choices=["全部", "HR", "Finance", "Technical", "General"],
                            value="全部",
                            label="分类"
                        )
                        user_id_input = gr.Textbox(
                            label="用户 ID",
                            value="default",
                            scale=1
                        )
                    
                    search_btn = gr.Button("🔍 搜索", variant="primary")
                
                with gr.Column(scale=2):
                    output = gr.Markdown(label="检索结果")
            
            search_btn.click(
                fn=search,
                inputs=[query_input, category_dropdown, user_id_input],
                outputs=output
            )
        
        return demo
    
    except ImportError:
        logger.warning("Gradio not installed, skipping UI")
        return None


# ============ 主函数 ============

def main():
    """主入口"""
    import uvicorn
    
    # 启动服务器
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7866,
        log_level="info"
    )


if __name__ == "__main__":
    main()
