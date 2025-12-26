"""
RAG Service
===========

Main RAG service that orchestrates the complete pipeline using LangGraph.
"""

import logging
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

# Handle optional LangGraph dependency
try:
    from langchain_ollama import ChatOllama
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    ChatOllama = None  # type: ignore
    StateGraph = None  # type: ignore
    END = None  # type: ignore

from config.settings import ModelConfig, RAGConfig
from core.models.state import GraphState
from core.models.response import RAGResponse, ConfidenceMetrics
from core.processors.query_processor import QueryProcessor, BaselineQueryProcessor
from core.processors.confidence_calculator import AdvancedConfidenceCalculator
from core.retrieval.vector_store import VectorStoreManager
from core.retrieval.reranker import CrossEncoderReranker
from core.retrieval.context_expander import ContextExpander
from core.generation.strict_generator import StrictGenerator
from core.generation.flexible_generator import FlexibleGenerator

logger = logging.getLogger(__name__)


class RAGService:
    """
    Main RAG service using LangGraph for pipeline orchestration.

    Handles the complete flow from query analysis to answer generation:
    1. Query Analysis
    2. Filter Planning
    3. MMR Retrieval
    4. Cross-Encoder Reranking
    5. Context Expansion
    6. Strict Generation
    7. Fallback to Flexible Generation (if needed)
    """

    def __init__(
        self,
        model_config: ModelConfig,
        rag_config: RAGConfig,
        debug_metadata: bool = False
    ):
        """
        Initialize RAG service with configuration.

        Args:
            model_config: Model configuration
            rag_config: RAG pipeline configuration
            debug_metadata: Whether to debug metadata fields
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph not available. "
                "Install with: pip install langgraph langchain-ollama"
            )

        self.model_config = model_config
        self.rag_config = rag_config
        self.debug_metadata = debug_metadata

        # Initialize components
        self._init_components()

        # Build LangGraph
        self.app = self._build_graph()

        # Statistics tracking
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "average_response_time": 0.0,
            "generation_modes": {"strict": 0, "flexible": 0},
            "model_name": model_config.ollama_model,
        }

        logger.info("RAGService initialized successfully")

    def _init_components(self) -> None:
        """Initialize all pipeline components."""
        # LLM
        self.llm = ChatOllama(model=self.model_config.ollama_model)

        # Vector store
        self.vector_store = VectorStoreManager(
            persist_dir=self.rag_config.persist_dir,
            collection_name=self.rag_config.collection_name,
            embed_model=self.model_config.embed_model,
            device=self.model_config.embed_device,
        )

        # Query processor
        self.query_processor = QueryProcessor()

        # Reranker
        self.reranker = CrossEncoderReranker(
            model_name=self.model_config.cross_encoder_model,
            device=self.model_config.cross_encoder_device,
            enabled=self.model_config.use_cross_encoder,
        )

        # Context expander
        self.context_expander = ContextExpander(
            token_budget=self.rag_config.token_budget
        )

        # Generators
        self.strict_generator = StrictGenerator(self.llm)
        self.flexible_generator = FlexibleGenerator(self.llm)

        # Confidence calculator
        self.confidence_calculator = AdvancedConfidenceCalculator(
            ollama_model=self.model_config.ollama_model
        )

    def _build_graph(self) -> Any:
        """Build the LangGraph processing pipeline."""
        graph = StateGraph(GraphState)

        # Add nodes
        graph.add_node("analyze_query", self._analyze_query)
        graph.add_node("plan_filters", self._plan_filters)
        graph.add_node("retrieve_mmr", self._retrieve_mmr)
        graph.add_node("rerank", self._rerank)
        graph.add_node("expand_context", self._expand_context)
        graph.add_node("generate_strict", self._generate_strict)
        graph.add_node("decide_fallback", self._decide_fallback)
        graph.add_node("generate_flexible", self._generate_flexible)

        # Wire the graph
        graph.set_entry_point("analyze_query")
        graph.add_edge("analyze_query", "plan_filters")
        graph.add_edge("plan_filters", "retrieve_mmr")
        graph.add_edge("retrieve_mmr", "rerank")
        graph.add_edge("rerank", "expand_context")
        graph.add_edge("expand_context", "generate_strict")
        graph.add_edge("generate_strict", "decide_fallback")

        # Conditional routing
        def route_decision(state: GraphState) -> str:
            return END if state.get("strict_ok") else "generate_flexible"

        graph.add_conditional_edges(
            "decide_fallback",
            route_decision,
            {END: END, "generate_flexible": "generate_flexible"}
        )
        graph.add_edge("generate_flexible", END)

        return graph.compile()

    # ==================== Graph Nodes ====================

    def _analyze_query(self, state: GraphState) -> Dict[str, Any]:
        """Node: Analyze query using query processor."""
        question = state["question"]

        try:
            result = self.query_processor.process_query(question)
            analysis = {
                "track": result.get_track(),
                "intent": result.intent.value if hasattr(result.intent, 'value') else str(result.intent),
                "entities": result.get_entity_dict(),
                "confidence": 0.8,
                "programs": result.programs,
                "page_types": result.page_types,
                "metadata_filters": result.metadata_filters,
            }
        except Exception as e:
            logger.warning(f"[analyze_query] Query processor failed: {e}")
            baseline = BaselineQueryProcessor()
            result = baseline.process_query(question)
            analysis = {
                "track": result.get_track(),
                "intent": result.intent.value,
                "entities": result.get_entity_dict(),
                "confidence": 0.5,
                "programs": result.programs,
                "page_types": result.page_types,
                "metadata_filters": result.metadata_filters,
            }

        telemetry = state.get("telemetry", {})
        telemetry["analysis"] = analysis

        logger.info(f"[analyze_query] track={analysis.get('track')} intent={analysis.get('intent')}")
        return {"analysis": analysis, "telemetry": telemetry}

    def _plan_filters(self, state: GraphState) -> Dict[str, Any]:
        """Node: Plan metadata filters based on analysis."""
        analysis = state["analysis"]

        # Use enhanced filters if available
        if analysis.get("metadata_filters"):
            where = analysis["metadata_filters"]
        else:
            # Build filters from analysis
            conditions = []

            programs = analysis.get("programs", [])
            if programs:
                if len(programs) == 1:
                    conditions.append({"program": {"$eq": programs[0]}})
                else:
                    conditions.append({"program": {"$in": programs}})

            page_types = analysis.get("page_types", [])
            if page_types:
                conditions.append({"page_type": {"$in": page_types}})

            if conditions:
                where = {"$and": conditions} if len(conditions) > 1 else conditions[0]
            else:
                where = None

        plan = {
            "where": where,
            "k": self.rag_config.k,
            "fetch_k": self.rag_config.fetch_k,
            "mmr_lambda": self.rag_config.mmr_lambda,
        }

        logger.info(f"[plan_filters] where={where}")
        return {"plan": plan}

    def _retrieve_mmr(self, state: GraphState) -> Dict[str, Any]:
        """Node: Retrieve documents using MMR."""
        plan = state["plan"]

        docs = self.vector_store.retrieve_mmr(
            query=state["question"],
            k=plan.get("k", self.rag_config.k),
            fetch_k=plan.get("fetch_k", self.rag_config.fetch_k),
            mmr_lambda=plan.get("mmr_lambda", self.rag_config.mmr_lambda),
            filters=plan.get("where"),
            debug_metadata=self.debug_metadata,
        )

        return {"retrieved_docs": docs}

    def _rerank(self, state: GraphState) -> Dict[str, Any]:
        """Node: Rerank documents using cross-encoder."""
        docs = state.get("retrieved_docs", [])

        reranked, scores = self.reranker.rerank(
            query=state["question"],
            documents=docs,
            top_n=self.rag_config.rerank_top_n,
        )

        return {"reranked_docs": reranked, "cross_encoder_scores": scores}

    def _expand_context(self, state: GraphState) -> Dict[str, Any]:
        """Node: Expand context within token budget."""
        docs = state.get("reranked_docs") or state.get("retrieved_docs") or []

        contexts, urls = self.context_expander.expand_context(
            documents=docs,
            max_contexts=self.rag_config.max_contexts,
        )

        return {"contexts": contexts, "used_urls": urls}

    def _generate_strict(self, state: GraphState) -> Dict[str, Any]:
        """Node: Generate answer in strict mode."""
        contexts = state.get("contexts", [])

        answer, success = self.strict_generator.generate(
            question=state["question"],
            contexts=contexts,
            urls=state.get("used_urls"),
        )

        return {"answer": answer, "strict_ok": success}

    def _decide_fallback(self, state: GraphState) -> Dict[str, Any]:
        """Node: Decide whether to use flexible generation."""
        answer = (state.get("answer") or "").strip()
        strict_ok = len(answer) >= self.rag_config.strict_min_length
        return {"strict_ok": strict_ok}

    def _generate_flexible(self, state: GraphState) -> Dict[str, Any]:
        """Node: Generate answer in flexible mode."""
        contexts = state.get("contexts", [])

        answer, success = self.flexible_generator.generate(
            question=state["question"],
            contexts=contexts,
            urls=state.get("used_urls"),
        )

        return {"answer": answer, "strict_ok": True}

    # ==================== Public API ====================

    def process_query(self, question: str) -> RAGResponse:
        """
        Process a query and return structured response.

        Args:
            question: User's question

        Returns:
            RAGResponse with answer, confidence, and metadata
        """
        start_time = time.time()
        self.stats["total_queries"] += 1

        try:
            # Initialize state
            init_state: GraphState = {
                "question": question,
                "telemetry": {},
            }

            # Run the graph
            final_state = self.app.invoke(init_state)

            # Extract results
            answer = (final_state.get("answer") or "").strip()
            used_urls = final_state.get("used_urls") or []
            analysis = final_state.get("analysis", {})
            retrieved_docs = final_state.get("retrieved_docs", [])
            reranked_docs = final_state.get("reranked_docs", [])
            cross_encoder_scores = final_state.get("cross_encoder_scores", [])
            strict_ok = final_state.get("strict_ok", False)
            contexts = final_state.get("contexts", [])

            # Determine generation mode
            generation_mode = "strict" if strict_ok else "flexible"
            self.stats["generation_modes"][generation_mode] += 1

            # Calculate confidence
            try:
                confidence_metrics = self.confidence_calculator.calculate_confidence(
                    query=question,
                    answer=answer,
                    retrieved_docs=retrieved_docs,
                    reranked_docs=reranked_docs,
                    cross_encoder_scores=cross_encoder_scores,
                    analysis=analysis,
                    generation_mode=generation_mode,
                )
                final_confidence = confidence_metrics.final_confidence
                confidence_metadata = {
                    "confidence_breakdown": confidence_metrics.to_dict(),
                    "llm_reasoning": confidence_metrics.reasoning,
                }
            except Exception as e:
                logger.error(f"Confidence calculation failed: {e}")
                final_confidence = 0.6
                confidence_metadata = {"error": str(e)}

            processing_time = time.time() - start_time

            # Build response
            response = RAGResponse(
                query=question,
                answer=answer or "I couldn't find a suitable answer for your question.",
                confidence=final_confidence,
                sources=used_urls[:10],
                generation_mode=generation_mode,
                processing_time=processing_time,
                reasoning_steps=[
                    f"Analysis: {analysis.get('intent', 'unknown')} query",
                    f"Documents retrieved: {len(retrieved_docs)}",
                    f"Documents reranked: {len(reranked_docs)}",
                    f"Contexts used: {len(contexts)}",
                ],
                conflicts_detected=[],
                contexts=contexts,
                retrieved_contexts=contexts,
                used_urls=used_urls,
                metadata={
                    "analysis": analysis,
                    "model_name": self.model_config.ollama_model,
                    "documents_retrieved": len(retrieved_docs),
                    "documents_reranked": len(reranked_docs),
                    "contexts_used": len(contexts),
                    **confidence_metadata,
                },
            )

            self.stats["successful_queries"] += 1
            self._update_average_response_time(processing_time)

            return response

        except Exception as e:
            logger.exception("Error processing query")
            processing_time = time.time() - start_time
            return RAGResponse.error_response(
                query=question,
                error=str(e),
                processing_time=processing_time,
                model_name=self.model_config.ollama_model,
            )

    def _update_average_response_time(self, new_time: float) -> None:
        """Update average response time statistics."""
        current = self.stats["average_response_time"]
        n = self.stats["successful_queries"]
        if n <= 1:
            self.stats["average_response_time"] = new_time
        else:
            self.stats["average_response_time"] = (current * (n - 1) + new_time) / n

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        total = max(1, self.stats["total_queries"])
        return {
            **self.stats,
            "success_rate": (self.stats["successful_queries"] / total) * 100.0,
            "timestamp": datetime.now().isoformat(),
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        try:
            self.llm.invoke("Test")
            llm_status = "available"
        except Exception:
            llm_status = "unavailable"

        try:
            self.vector_store.similarity_search("test", k=1)
            vs_status = "available"
        except Exception:
            vs_status = "unavailable"

        return {
            "status": "healthy" if llm_status == "available" and vs_status == "available" else "degraded",
            "components": {
                "llm": llm_status,
                "vector_store": vs_status,
                "cross_encoder": "available" if self.reranker.is_available else "disabled",
                "query_processor": "available",
            },
            "model_name": self.model_config.ollama_model,
            "timestamp": datetime.now().isoformat(),
        }

    def get_system_info(self) -> Dict[str, Any]:
        """Get system configuration information."""
        return {
            "model": self.model_config.ollama_model,
            "embedding_model": self.model_config.embed_model,
            "collection": self.rag_config.collection_name,
            "cross_encoder_enabled": self.reranker.is_available,
            "token_budget": self.rag_config.token_budget,
            "version": "2.0.0",
        }
