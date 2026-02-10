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
        num_contexts: int = 4,  # ✅ ADDED
        conversation_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process user query with agentic reasoning.
        
        Args:
            user_query: The user's question
            num_contexts: Number of context chunks to retrieve (default: 4)
            conversation_history: Previous messages for context
            
        Returns:
            Dict with answer, sources, and metadata
        """
        
        # Store for use in tools
        self.default_num_contexts = num_contexts  # ✅ ADDED
        
        # Build conversation context
        messages = conversation_history or []
        messages.append({
            "role": "user",
            "content": user_query
        })
        
        # Initial agent prompt
        system_prompt = """You are Study Buddy, a friendly and encouraging AI tutor. You help students learn from their uploaded study materials.

        IMPORTANT: The user has already uploaded study materials. They are available in your search tools. DO NOT ask the user to provide materials - they're already here!

        Your personality:
        - Warm and encouraging, like a patient teacher
        - Enthusiastic about learning
        - Celebrates student progress
        - Uses casual, friendly language
        - Remembers what you discussed earlier in the conversation

        Available tools:
        1. **search_documents**: Search study materials for information
        2. **multi_search**: Compare multiple topics
        3. **generate_practice_questions**: Create practice questions from uploaded materials

        When the user says things like:
        - "test me" / "quiz me" / "practice questions" → Use generate_practice_questions tool
        - "explain X" / "what is X" → Use search_documents tool
        - "compare X and Y" / "difference between X and Y" → Use multi_search tool

        CRITICAL: The materials are already uploaded and ready to search. Never ask the user to provide materials.

        Call the appropriate tool now to help the student."""

        print(f"\n🤖 Agent processing: '{user_query}'")
        
        # Agent reasoning loop
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            temperature=0.3,
            system=system_prompt,
            tools=self.tools,
            tool_choice={"type": "any"},
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
                        num_results=tool_input.get("num_results", None)
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
                    # Practice questions tool returns sources too
                    if "sources" in result:
                        sources.extend(result["sources"])
        
        # ✅ NEW: Better synthesis logic
        if tool_results and sources:
            # We have search results - build context from chunks
            all_chunks = []
            for result in tool_results:
                if "chunks" in result:
                    all_chunks.extend(result["chunks"])
            
            # Build context from chunk texts
            context_parts = []
            for i, chunk in enumerate(all_chunks[:8], 1):  # Use top 8 chunks
                context_parts.append(f"[Source {i} - {chunk['filename']}]\n{chunk['text']}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Create a proper RAG-style prompt
            system_prompt = """You are a helpful study assistant. Answer questions based on the provided context from the user's study materials.

    Guidelines:
    - Use only information from the provided context
    - Be clear and educational
    - Structure your answer with headers if comparing multiple things
    - Don't mention that you're looking at sources - just provide the answer"""

            user_prompt = f"""Context from study materials:

    {context}

    ---

    Question: {original_query}

    Please provide a clear, well-structured answer based on the context above."""

            # Generate final answer
            synthesis_response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                temperature=0.3,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            final_answer = synthesis_response.content[0].text
            print(f"✅ Synthesized answer from {len(all_chunks)} chunks")
        
        elif tool_results:
            # For non-search tools (like practice questions)
            final_answer = tool_results[0].get("questions", final_answer)
        
        # If no tools were used, final_answer already has the text response
        
        return {
            "answer": final_answer,
            "sources": sources,
            "tool_calls": len(tool_results)
        }
    
    async def _search_documents(self, query: str, num_results: int = None) -> Dict:
        """Search documents using RAG."""
        
        # Use provided num_results, or fall back to default
        if num_results is None:
            num_results = getattr(self, 'default_num_contexts', 4)  # ✅ ADDED
        
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
        source_references = []  # ✅ ADD THIS
        
        for result in tool_results:
            if result["tool"] == "search_documents":
                context_parts.append(f"Search results for '{result['query']}':")
                for i, chunk in enumerate(result["chunks"], 1):
                    context_parts.append(f"[Source {i}] {chunk['text']}")  # ✅ CHANGED
                    source_references.append(f"Source {i}: {chunk['filename']}")  # ✅ ADD
            
            elif result["tool"] == "multi_search":
                context_parts.append("Comparison search results:")
                source_num = 1
                for query in result["queries"]:
                    context_parts.append(f"\nFor '{query}':")
                    # Find chunks for this query
                    relevant_chunks = [c for c in result["chunks"] if query.lower() in c.get('text', '').lower()][:2]
                    for chunk in relevant_chunks:
                        context_parts.append(f"[Source {source_num}] {chunk['text'][:300]}")
                        source_references.append(f"Source {source_num}: {chunk['filename']}")
                        source_num += 1
            
            elif result["tool"] == "generate_practice_questions":
                context_parts.append(result["questions"])
        
        context = "\n\n".join(context_parts)
        
        # Final synthesis with source attribution
        synthesis_prompt = f"""Original question: {original_query}

    Information gathered from study materials:
    {context}

    Please provide a comprehensive, clear answer to the original question. When stating facts, reference which source number supports that information (e.g., "According to Source 2..."). Be educational and encouraging."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": synthesis_prompt}]
        )
        
        answer = response.content[0].text
        
        # ADD source list at the end
        if source_references:
            answer += "\n\n**Sources used:**\n" + "\n".join(f"- {ref}" for ref in source_references[:5])
        
        return answer