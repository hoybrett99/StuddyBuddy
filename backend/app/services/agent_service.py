from typing import List, Dict, Any, Optional
from anthropic import Anthropic
import json

class StudyBuddyAgent:
    """
    AI Agent that intelligently handles user queries.
    
    Capabilities:
    - Intent detection (question type)
    - Multi-step reasoning
    - Tool usage (search, summarize, generate)
    - Conversation management
    """
    
    def __init__(self, claude_api_key: str, rag_service):
        self.client = Anthropic(api_key=claude_api_key)
        self.rag_service = rag_service
        
        # Define agent tools
        self.tools = [
            {
                "name": "search_documents",
                "description": "Search through uploaded study materials for relevant information. Use this for factual questions, definitions, and explanations.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query (be specific and focused)"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of relevant chunks to retrieve (default 4, use 6-8 for complex questions)"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "multi_search",
                "description": "Perform multiple searches for multi-part questions or comparisons. Use when the question asks about several different topics.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of focused search queries"
                        }
                    },
                    "required": ["queries"]
                }
            },
            {
                "name": "generate_practice_questions",
                "description": "Generate practice questions based on document content. Use when user asks for test questions, practice, or self-assessment.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Topic to generate questions about"
                        },
                        "num_questions": {
                            "type": "integer",
                            "description": "Number of questions to generate"
                        }
                    },
                    "required": ["topic"]
                }
            }
        ]
    
    async def process_query(
        self, 
        user_query: str,
        conversation_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process user query with agentic reasoning.
        
        Args:
            user_query: The user's question
            conversation_history: Previous messages for context
            
        Returns:
            Dict with answer, sources, and metadata
        """
        
        # Build conversation context
        messages = conversation_history or []
        messages.append({
            "role": "user",
            "content": user_query
        })
        
        # Initial agent prompt
        system_prompt = """You are Study Buddy, an intelligent AI tutor assistant. Your job is to help students learn from their uploaded study materials.

Your capabilities:
1. **search_documents**: Find specific information in study materials
2. **multi_search**: Search multiple topics for comparison questions
3. **generate_practice_questions**: Create test questions from content

Guidelines:
- For simple questions: use search_documents with a clear query
- For "what's the difference between X and Y": use multi_search with separate queries for X and Y
- For broad questions: break into focused sub-queries
- For practice/test questions: use generate_practice_questions
- Always explain your reasoning before calling tools
- Be encouraging and educational

Analyze the user's question and decide which tool(s) to use."""

        print(f"\n🤖 Agent processing: '{user_query}'")
        
        # Agent reasoning loop
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            temperature=0.3,
            system=system_prompt,
            tools=self.tools,
            messages=messages
        )
        
        # Process agent's decision
        return await self._execute_tools(response, user_query)
    
    async def _execute_tools(self, response, original_query: str) -> Dict[str, Any]:
        """Execute tools based on agent's decision."""
        
        final_answer = ""
        sources = []
        tool_results = []
        
        # Check what the agent wants to do
        for block in response.content:
            if block.type == "text":
                final_answer += block.text
            
            elif block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                
                print(f"  🔧 Using tool: {tool_name}")
                print(f"     Input: {tool_input}")
                
                # Execute the tool
                if tool_name == "search_documents":
                    result = await self._search_documents(
                        query=tool_input["query"],
                        num_results=tool_input.get("num_results", 4)
                    )
                    tool_results.append(result)
                    sources.extend(result["sources"])
                
                elif tool_name == "multi_search":
                    result = await self._multi_search(tool_input["queries"])
                    tool_results.append(result)
                    sources.extend(result["sources"])
                
                elif tool_name == "generate_practice_questions":
                    result = await self._generate_practice_questions(
                        topic=tool_input["topic"],
                        num_questions=tool_input.get("num_questions", 5)
                    )
                    tool_results.append(result)
        
        # If agent used tools, get final synthesis
        if tool_results:
            final_answer = await self._synthesize_answer(
                original_query=original_query,
                tool_results=tool_results
            )
        
        return {
            "answer": final_answer,
            "sources": sources,
            "tool_calls": len(tool_results)
        }
    
    async def _search_documents(self, query: str, num_results: int = 4) -> Dict:
        """Search documents using RAG."""
        chunks = await self.rag_service.search_similar_chunks(
            query=query,
            num_results=num_results
        )
        
        return {
            "tool": "search_documents",
            "query": query,
            "chunks": chunks,
            "sources": [
                {
                    "document_name": c["filename"],
                    "chunk_id": c["chunk_id"],
                    "relevance_score": c["similarity"],
                    "chunk_text": c["text"]
                }
                for c in chunks
            ]
        }
    
    async def _multi_search(self, queries: List[str]) -> Dict:
        """Perform multiple searches."""
        all_chunks = []
        sources = []
        
        for query in queries:
            result = await self._search_documents(query, num_results=3)
            all_chunks.extend(result["chunks"])
            sources.extend(result["sources"])
        
        return {
            "tool": "multi_search",
            "queries": queries,
            "chunks": all_chunks,
            "sources": sources
        }
    
    async def _generate_practice_questions(self, topic: str, num_questions: int = 5) -> Dict:
        """Generate practice questions based on topic."""
        
        # First, search for content on the topic
        search_result = await self._search_documents(topic, num_results=6)
        
        if not search_result["chunks"]:
            return {
                "tool": "generate_practice_questions",
                "questions": [],
                "error": "No content found on this topic"
            }
        
        # Build context from chunks
        context = "\n\n".join([c["text"] for c in search_result["chunks"]])
        
        # Generate questions
        prompt = f"""Based on this study material about {topic}, generate {num_questions} practice questions that test understanding.

Material:
{context}

Generate questions in this format:
1. [Question]
   Answer: [Brief answer]

Make questions varied (definitions, explanations, comparisons, applications)."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        
        questions = response.content[0].text
        
        return {
            "tool": "generate_practice_questions",
            "topic": topic,
            "questions": questions,
            "sources": search_result["sources"]
        }
    
    async def _synthesize_answer(self, original_query: str, tool_results: List[Dict]) -> str:
        """Synthesize final answer from tool results."""
        
        # Build context from all tool results
        context_parts = []
        
        for result in tool_results:
            if result["tool"] == "search_documents":
                context_parts.append(f"Search results for '{result['query']}':")
                for chunk in result["chunks"]:
                    context_parts.append(f"- {chunk['text'][:200]}...")
            
            elif result["tool"] == "multi_search":
                for i, query in enumerate(result["queries"]):
                    context_parts.append(f"Results for '{query}':")
                    # Show relevant chunks
            
            elif result["tool"] == "generate_practice_questions":
                context_parts.append(result["questions"])
        
        context = "\n\n".join(context_parts)
        
        # Final synthesis
        synthesis_prompt = f"""Original question: {original_query}

Information gathered:
{context}

Please provide a comprehensive, clear answer to the original question based on this information. Be educational and encouraging."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": synthesis_prompt}]
        )
        
        return response.content[0].text