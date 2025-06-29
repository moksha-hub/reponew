# RepoMaster: Autonomous Exploration and Understanding of GitHub Repositories
# Complete implementation based on the paper and README

import os
import sys
import json
import asyncio
import logging
import subprocess
import requests
import time
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import ast
import re
from collections import defaultdict, deque
import networkx as nx
from datetime import datetime
import yaml
import pickle
import aiohttp
import aiofiles

# Core dependencies
try:
    import openai
    import anthropic
    from github import Github
    import tiktoken
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss
    from sklearn.metrics.pairwise import cosine_similarity
    import gitpython as git
    from git import Repo
    import tree_sitter
    from tree_sitter import Language, Parser
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install required packages:")
    print("pip install openai anthropic PyGithub tiktoken numpy sentence-transformers faiss-cpu scikit-learn gitpython tree-sitter aiohttp aiofiles")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration and Data Structures
# ==============================================================================

@dataclass
class LLMConfig:
    """Configuration for LLM backend"""
    model: str = "claude-3-5-sonnet-20241022"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout: int = 2000

@dataclass
class CodeExecutionConfig:
    """Configuration for code execution environment"""
    work_dir: str = "workspace"
    use_docker: bool = False
    timeout: int = 7200
    use_venv: bool = True

@dataclass
class ExplorerConfig:
    """Configuration for repository exploration"""
    max_turns: int = 40
    function_call: bool = True
    repo_init: bool = True
    max_context_length: int = 8000
    compression_ratio: float = 0.3

@dataclass
class RepositoryInfo:
    """Information about a repository"""
    name: str
    url: str
    description: str
    stars: int
    language: str
    topics: List[str]
    readme_content: str
    relevance_score: float = 0.0

@dataclass
class CodeNode:
    """Node in the Hierarchical Code Tree (HCT)"""
    name: str
    type: str  # 'file', 'class', 'function', 'variable'
    path: str
    content: str
    start_line: int
    end_line: int
    parent: Optional['CodeNode'] = None
    children: List['CodeNode'] = field(default_factory=list)
    importance_score: float = 0.0

@dataclass
class FunctionCall:
    """Function call in the Function Call Graph (FCG)"""
    caller: str
    callee: str
    call_type: str  # 'direct', 'indirect', 'dynamic'
    line_number: int
    confidence: float = 1.0

@dataclass
class ModuleDependency:
    """Module dependency in the Module Dependency Graph (MDG)"""
    source: str
    target: str
    dependency_type: str  # 'import', 'from_import', 'dynamic_import'
    line_number: int

# ==============================================================================
# LLM Backend Abstraction
# ==============================================================================

class LLMBackend:
    """Abstract base class for LLM backends"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        self._setup_client()

    def _setup_client(self):
        """Setup the LLM client based on model type"""
        if "gpt" in self.config.model.lower():
            self.client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url
            )
            self.backend_type = "openai"
        elif "claude" in self.config.model.lower():
            self.client = anthropic.Anthropic(
                api_key=self.config.api_key,
                base_url=self.config.base_url
            )
            self.backend_type = "anthropic"
        elif "deepseek" in self.config.model.lower():
            self.client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url or "https://api.deepseek.com"
            )
            self.backend_type = "deepseek"
        else:
            raise ValueError(f"Unsupported model: {self.config.model}")

    async def generate_response(self, messages: List[Dict], **kwargs) -> str:
        """Generate response from LLM"""
        try:
            if self.backend_type == "anthropic":
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=messages,
                    **kwargs
                )
                return response.content[0].text
            else:  # OpenAI-compatible
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    **kwargs
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"Error generating response: {str(e)}"

# ==============================================================================
# Repository Search Module
# ==============================================================================

class RepositorySearcher:
    """Intelligent repository search with multi-round optimization"""

    def __init__(self, github_token: Optional[str] = None):
        self.github = Github(github_token) if github_token else Github()
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    def extract_keywords(self, task_description: str) -> List[str]:
        """Extract relevant keywords from task description"""
        import re
        words = re.findall(r'\b\w+\b', task_description.lower())
        
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:10]

    def generate_search_queries(self, task_description: str) -> List[str]:
        """Generate multiple search queries for comprehensive coverage"""
        keywords = self.extract_keywords(task_description)
        
        queries = []
        
        if keywords:
            queries.append(' '.join(keywords[:3]))
        
        if any(word in task_description.lower() for word in ['image', 'photo', 'picture']):
            queries.extend(['image processing', 'computer vision', 'PIL', 'opencv'])
        
        if any(word in task_description.lower() for word in ['pdf', 'document']):
            queries.extend(['pdf processing', 'document extraction', 'pypdf'])
        
        if any(word in task_description.lower() for word in ['neural', 'deep', 'machine learning', 'ai']):
            queries.extend(['deep learning', 'neural network', 'pytorch', 'tensorflow'])
        
        if any(word in task_description.lower() for word in ['video', 'movie']):
            queries.extend(['video processing', 'ffmpeg', 'opencv video'])
        
        if any(word in task_description.lower() for word in ['style', 'transfer', 'art']):
            queries.extend(['neural style transfer', 'artistic style', 'style transfer'])
        
        return list(set(queries))[:5]

    async def search_repositories(self, task_description: str, max_repos: int = 20) -> List[RepositoryInfo]:
        """Search for relevant repositories"""
        queries = self.generate_search_queries(task_description)
        all_repos = []
        
        for query in queries:
            try:
                repos = self.github.search_repositories(
                    query=query,
                    sort='stars',
                    order='desc'
                )
                
                for repo in repos[:max_repos // len(queries)]:
                    repo_info = RepositoryInfo(
                        name=repo.full_name,
                        url=repo.clone_url,
                        description=repo.description or "",
                        stars=repo.stargazers_count,
                        language=repo.language or "Unknown",
                        topics=repo.get_topics(),
                        readme_content=self._get_readme_content(repo)
                    )
                    all_repos.append(repo_info)
                    
            except Exception as e:
                logger.warning(f"Search error for query '{query}': {e}")
                continue
        
        unique_repos = {repo.name: repo for repo in all_repos}.values()
        ranked_repos = self._rank_repositories(list(unique_repos), task_description)
        
        return ranked_repos[:max_repos]

    def _get_readme_content(self, repo) -> str:
        """Get README content from repository"""
        try:
            readme = repo.get_readme()
            return readme.decoded_content.decode('utf-8')[:2000]
        except:
            return ""

    def _rank_repositories(self, repos: List[RepositoryInfo], task_description: str) -> List[RepositoryInfo]:
        """Rank repositories by relevance to task"""
        if not repos:
            return repos
        
        repo_texts = [f"{repo.description} {repo.readme_content}" for repo in repos]
        
        task_embedding = self.sentence_model.encode([task_description])
        repo_embeddings = self.sentence_model.encode(repo_texts)
        
        similarities = cosine_similarity(task_embedding, repo_embeddings)[0]
        
        for i, repo in enumerate(repos):
            semantic_score = similarities[i]
            star_score = min(repo.stars / 10000, 1.0)
            topic_score = self._calculate_topic_relevance(repo.topics, task_description)
            
            repo.relevance_score = 0.5 * semantic_score + 0.3 * star_score + 0.2 * topic_score
        
        return sorted(repos, key=lambda x: x.relevance_score, reverse=True)

    def _calculate_topic_relevance(self, topics: List[str], task_description: str) -> float:
        """Calculate relevance based on repository topics"""
        if not topics:
            return 0.0
        
        task_words = set(task_description.lower().split())
        topic_words = set(' '.join(topics).lower().split())
        
        intersection = task_words.intersection(topic_words)
        union = task_words.union(topic_words)
        
        return len(intersection) / len(union) if union else 0.0

# ==============================================================================
# Code Structure Analysis
# ==============================================================================

class CodeStructureAnalyzer:
    """Analyze code structure and build HCT, FCG, and MDG"""
    
    def __init__(self):
        self.hct = None
        self.fcg = nx.DiGraph()
        self.mdg = nx.DiGraph()
        self.importance_scores = {}
    
    def analyze_repository(self, repo_path: str) -> Dict[str, Any]:
        """Analyze repository structure and build graphs"""
        logger.info(f"Analyzing repository structure: {repo_path}")
        
        self.hct = self._build_hct(repo_path)
        self._build_fcg(repo_path)
        self._build_mdg(repo_path)
        self._calculate_importance_scores()
        
        return {
            'hct': self.hct,
            'fcg': self.fcg,
            'mdg': self.mdg,
            'importance_scores': self.importance_scores
        }
    
    def _build_hct(self, repo_path: str) -> CodeNode:
        """Build Hierarchical Code Tree"""
        root = CodeNode(
            name=os.path.basename(repo_path),
            type='directory',
            path=repo_path,
            content='',
            start_line=0,
            end_line=0
        )
        
        self._process_directory(repo_path, root)
        return root
    
    def _process_directory(self, dir_path: str, parent_node: CodeNode):
        """Recursively process directory and build tree"""
        try:
            for item in os.listdir(dir_path):
                if item.startswith('.'):
                    continue
                
                item_path = os.path.join(dir_path, item)
                
                if os.path.isdir(item_path):
                    dir_node = CodeNode(
                        name=item,
                        type='directory',
                        path=item_path,
                        content='',
                        start_line=0,
                        end_line=0,
                        parent=parent_node
                    )
                    parent_node.children.append(dir_node)
                    self._process_directory(item_path, dir_node)
                
                elif item.endswith(('.py', '.js', '.java', '.cpp', '.c', '.h')):
                    file_node = self._process_file(item_path, parent_node)
                    if file_node:
                        parent_node.children.append(file_node)
        
        except PermissionError:
            logger.warning(f"Permission denied: {dir_path}")

    def _process_file(self, file_path: str, parent_node: CodeNode) -> Optional[CodeNode]:
        """Process a single file and extract structure"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            file_node = CodeNode(
                name=os.path.basename(file_path),
                type='file',
                path=file_path,
                content=content[:1000],
                start_line=1,
                end_line=len(content.splitlines()),
                parent=parent_node
            )
            
            if file_path.endswith('.py'):
                self._parse_python_file(content, file_node)
            
            return file_node
            
        except Exception as e:
            logger.warning(f"Error processing file {file_path}: {e}")
            return None

    def _parse_python_file(self, content: str, file_node: CodeNode):
        """Parse Python file for classes and functions"""
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_node = CodeNode(
                        name=node.name,
                        type='class',
                        path=file_node.path,
                        content=ast.get_source_segment(content, node) or '',
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        parent=file_node
                    )
                    file_node.children.append(class_node)
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_node = CodeNode(
                                name=item.name,
                                type='method',
                                path=file_node.path,
                                content=ast.get_source_segment(content, item) or '',
                                start_line=item.lineno,
                                end_line=item.end_lineno or item.lineno,
                                parent=class_node
                            )
                            class_node.children.append(method_node)
                
                elif isinstance(node, ast.FunctionDef) and not isinstance(node, ast.ClassDef):
                    func_node = CodeNode(
                        name=node.name,
                        type='function',
                        path=file_node.path,
                        content=ast.get_source_segment(content, node) or '',
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        parent=file_node
                    )
                    file_node.children.append(func_node)
        
        except SyntaxError:
            logger.warning(f"Syntax error in Python file: {file_node.path}")

    def _build_fcg(self, repo_path: str):
        """Build Function Call Graph"""
        python_files = []
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        for file_path in python_files:
            self._analyze_function_calls(file_path)

    def _analyze_function_calls(self, file_path: str):
        """Analyze function calls in a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            current_function = None
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    current_function = f"{file_path}:{node.name}"
                
                elif isinstance(node, ast.Call) and current_function:
                    callee = self._extract_call_name(node)
                    if callee:
                        self.fcg.add_edge(
                            current_function,
                            callee,
                            line_number=node.lineno,
                            call_type='direct'
                        )
        
        except Exception as e:
            logger.warning(f"Error analyzing function calls in {file_path}: {e}")

    def _extract_call_name(self, call_node: ast.Call) -> Optional[str]:
        """Extract function name from call node"""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            return call_node.func.attr
        return None
    
    def _build_mdg(self, repo_path: str):
        """Build Module Dependency Graph"""
        python_files = []
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        for file_path in python_files:
            self._analyze_imports(file_path)

    def _analyze_imports(self, file_path: str):
        """Analyze imports in a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.mdg.add_edge(
                            file_path,
                            alias.name,
                            dependency_type='import',
                            line_number=node.lineno
                        )
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self.mdg.add_edge(
                            file_path,
                            node.module,
                            dependency_type='from_import',
                            line_number=node.lineno
                        )
        
        except Exception as e:
            logger.warning(f"Error analyzing imports in {file_path}: {e}")

    def _calculate_importance_scores(self):
        """Calculate importance scores for code components"""
        if self.fcg.nodes():
            pagerank_scores = nx.pagerank(self.fcg)
            self.importance_scores.update(pagerank_scores)
        
        if self.mdg.nodes():
            centrality_scores = nx.degree_centrality(self.mdg)
            for node, score in centrality_scores.items():
                self.importance_scores[node] = self.importance_scores.get(node, 0) + score
    
    def get_core_components(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get top-k core components by importance"""
        sorted_components = sorted(
            self.importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_components[:top_k]

# ==============================================================================
# Context Management and Compression
# ==============================================================================

class ContextManager:
    """Manage and compress code context for LLM"""
    
    def __init__(self, max_context_length: int = 8000):
        self.max_context_length = max_context_length
        self.current_context = {}
        self.context_history = []
    
    def initialize_context(self, repo_info: RepositoryInfo, analysis_result: Dict) -> str:
        """Initialize context with repository overview"""
        context_parts = []
        
        context_parts.append(f"# Repository: {repo_info.name}")
        context_parts.append(f"Description: {repo_info.description}")
        context_parts.append(f"Language: {repo_info.language}")
        context_parts.append(f"Stars: {repo_info.stars}")
        context_parts.append(f"Topics: {', '.join(repo_info.topics)}")
        context_parts.append("")
        
        if repo_info.readme_content:
            context_parts.append("## README Summary")
            context_parts.append(self._summarize_text(repo_info.readme_content, 500))
            context_parts.append("")
        
        if 'importance_scores' in analysis_result:
            context_parts.append("## Core Components")
            analyzer = CodeStructureAnalyzer()
            analyzer.importance_scores = analysis_result['importance_scores']
            core_components = analyzer.get_core_components(5)
            
            for component, score in core_components:
                context_parts.append(f"- {component} (importance: {score:.3f})")
            context_parts.append("")
        
        if 'hct' in analysis_result:
            context_parts.append("## Directory Structure")
            context_parts.append(self._format_directory_tree(analysis_result['hct'], max_depth=3))
            context_parts.append("")
        
        initial_context = "\n".join(context_parts)
        return self._compress_context(initial_context)

    def _summarize_text(self, text: str, max_length: int) -> str:
        """Summarize text to fit within max_length"""
        if len(text) <= max_length:
            return text
        
        return text[:max_length-3] + "..."

    def _format_directory_tree(self, root: CodeNode, max_depth: int = 3, current_depth: int = 0) -> str:
        """Format directory tree structure"""
        if current_depth > max_depth:
            return ""
        
        lines = []
        indent = "  " * current_depth
        
        if root.type == 'directory':
            lines.append(f"{indent}{root.name}/")
        else:
            lines.append(f"{indent}{root.name}")
        
        for child in root.children[:10]:
            child_lines = self._format_directory_tree(child, max_depth, current_depth + 1)
            if child_lines:
                lines.append(child_lines)
        
        return "\n".join(lines)

    def _compress_context(self, context: str) -> str:
        """Compress context to fit within max_context_length"""
        if len(context) <= self.max_context_length:
            return context
        
        return context[:self.max_context_length-100] + "\n... (content truncated)"

    def add_code_view(self, file_path: str, content: str, view_type: str = "file") -> str:
        """Add code view to context"""
        view_header = f"\n## {view_type.title()} View: {file_path}\n"
        view_content = f"```python\n{content}\n```\n"
        
        new_content = view_header + view_content
        
        if len(self.current_context.get('main', '')) + len(new_content) > self.max_context_length:
            self.current_context['main'] = self._compress_context(
                self.current_context.get('main', '') + new_content
            )
        else:
            self.current_context['main'] = self.current_context.get('main', '') + new_content
        
        return self.current_context['main']

# ==============================================================================
# Code Exploration Tools
# ==============================================================================

class CodeExplorer:
    """Core code exploration and analysis tool"""

    def __init__(self,
                 repo_path: str,
                 work_dir: str,
                 llm_config: LLMConfig,
                 explorer_config: ExplorerConfig):
        self.repo_path = repo_path
        self.work_dir = work_dir
        self.llm_config = llm_config
        self.explorer_config = explorer_config
        
        self.llm = LLMBackend(llm_config)
        self.analyzer = CodeStructureAnalyzer()
        self.context_manager = ContextManager(explorer_config.max_context_length)
        
        self.analysis_result = None
        self.current_context = ""
        self.conversation_history = []

    async def initialize_exploration(self, repo_info: RepositoryInfo) -> str:
        """Initialize repository exploration"""
        logger.info(f"Initializing exploration for {repo_info.name}")
        
        self.analysis_result = self.analyzer.analyze_repository(self.repo_path)
        
        self.current_context = self.context_manager.initialize_context(
            repo_info, self.analysis_result
        )
        
        return self.current_context

    async def explore_code(self, query: str) -> str:
        """Explore code based on query"""
        tools = {
            "view_file": self._view_file,
            "view_class": self._view_class,
            "view_function": self._view_function,
            "search_code": self._search_code,
            "analyze_dependencies": self._analyze_dependencies,
            "trace_execution": self._trace_execution,
            "list_directory": self._list_directory
        }
        
        tool_name = await self._select_exploration_tool(query)
        
        if tool_name in tools:
            result = await tools[tool_name](query)
            return result
        else:
            return f"Unknown exploration tool: {tool_name}"

    async def _select_exploration_tool(self, query: str) -> str:
        """Select appropriate exploration tool based on query"""
        system_prompt = """
        You are a code exploration assistant. Based on the user's query, select the most appropriate exploration tool:
        
        Available tools:
        - view_file: View complete file content
        - view_class: View specific class definition
        - view_function: View specific function definition
        - search_code: Search for code patterns or keywords
        - analyze_dependencies: Analyze module dependencies
        - trace_execution: Trace function call paths
        - list_directory: List directory contents
        
        Respond with only the tool name.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"}
        ]
        
        response = await self.llm.generate_response(messages)
        return response.strip().lower()

    async def _view_file(self, file_path: str) -> str:
        """View complete file content"""
        try:
            full_path = os.path.join(self.repo_path, file_path.strip())
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            self.context_manager.add_code_view(file_path, content, "file")
            
            return f"File content loaded: {file_path}\n```python\n{content[:2000]}\n```"
        
        except Exception as e:
            return f"Error viewing file {file_path}: {str(e)}"

    async def _view_class(self, query: str) -> str:
        """View specific class definition"""
        class_name = self._extract_identifier(query, "class")
        
        if not class_name:
            return "Could not extract class name from query"
        
        class_node = self._find_code_node(self.analysis_result['hct'], class_name, 'class')
        
        if class_node:
            self.context_manager.add_code_view(class_node.path, class_node.content, "class")
            return f"Class {class_name} found:\n```python\n{class_node.content}\n```"
        else:
            return f"Class {class_name} not found"

    async def _view_function(self, query: str) -> str:
        """View specific function definition"""
        func_name = self._extract_identifier(query, "function")
        
        if not func_name:
            return "Could not extract function name from query"
        
        func_node = self._find_code_node(self.analysis_result['hct'], func_name, 'function')
        
        if func_node:
            self.context_manager.add_code_view(func_node.path, func_node.content, "function")
            return f"Function {func_name} found:\n```python\n{func_node.content}\n```"
        else:
            return f"Function {func_name} not found"

    async def _search_code(self, query: str) -> str:
        """Search for code patterns or keywords"""
        search_terms = self._extract_search_terms(query)
        results = []
        
        for root, dirs, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith(('.py', '.js', '.java', '.cpp', '.c', '.h')):
                    file_path = os.path.join(root, file)
                    matches = self._search_in_file(file_path, search_terms)
                    if matches:
                        results.extend(matches)
        
        if results:
            return self._format_search_results(results[:10])
        else:
            return f"No matches found for: {search_terms}"

    async def _analyze_dependencies(self, query: str) -> str:
        """Analyze module dependencies"""
        if not self.analysis_result or 'mdg' not in self.analysis_result:
            return "Module dependency graph not available"
        
        mdg = self.analysis_result['mdg']
        
        stats = {
            'total_modules': len(mdg.nodes()),
            'total_dependencies': len(mdg.edges()),
            'most_dependent': self._get_most_dependent_modules(mdg),
            'dependency_cycles': self._detect_dependency_cycles(mdg)
        }
        
        return self._format_dependency_analysis(stats)

    async def _trace_execution(self, query: str) -> str:
        """Trace function call paths"""
        if not self.analysis_result or 'fcg' not in self.analysis_result:
            return "Function call graph not available"
        
        func_name = self._extract_identifier(query, "function")
        if not func_name:
            return "Could not extract function name for tracing"
        
        fcg = self.analysis_result['fcg']
        
        paths = self._find_execution_paths(fcg, func_name)
        
        return self._format_execution_paths(paths)

    async def _list_directory(self, query: str) -> str:
        """List directory contents"""
        dir_path = self._extract_path(query) or ""
        full_path = os.path.join(self.repo_path, dir_path)
        
        if not os.path.exists(full_path):
            return f"Directory not found: {dir_path}"
        
        try:
            items = []
            for item in os.listdir(full_path):
                item_path = os.path.join(full_path, item)
                if os.path.isdir(item_path):
                    items.append(f"📁 {item}/")
                else:
                    items.append(f"📄 {item}")
            
            return f"Directory listing for {dir_path or 'root'}:\n" + "\n".join(items[:20])
        
        except PermissionError:
            return f"Permission denied: {dir_path}"

    def _extract_identifier(self, query: str, identifier_type: str) -> Optional[str]:
        """Extract identifier (class, function, etc.) from query"""
        import re
        
        patterns = {
            'class': r'class\s+(\w+)|(\w+)\s+class',
            'function': r'function\s+(\w+)|(\w+)\s+function|def\s+(\w+)',
            'method': r'method\s+(\w+)|(\w+)\s+method'
        }
        
        pattern = patterns.get(identifier_type, r'(\w+)')
        match = re.search(pattern, query, re.IGNORECASE)
        
        if match:
            return next((group for group in match.groups() if group), None)
        
        return None

    def _find_code_node(self, root: CodeNode, name: str, node_type: str) -> Optional[CodeNode]:
        """Find code node by name and type in HCT"""
        if root.name == name and root.type == node_type:
            return root
        
        for child in root.children:
            result = self._find_code_node(child, name, node_type)
            if result:
                return result
        
        return None

    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract search terms from query"""
        stop_words = {'search', 'find', 'look', 'for', 'code', 'function', 'class', 'method'}
        words = query.lower().split()
        return [word for word in words if word not in stop_words and len(word) > 2]

    def _search_in_file(self, file_path: str, search_terms: List[str]) -> List[Dict]:
        """Search for terms in a file"""
        matches = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                for term in search_terms:
                    if term.lower() in line.lower():
                        matches.append({
                            'file': file_path,
                            'line': line_num,
                            'content': line.strip(),
                            'term': term
                        })
        except Exception:
            pass
        
        return matches

    def _format_search_results(self, results: List[Dict]) -> str:
        """Format search results for display"""
        output = ["Search Results:"]
        
        for result in results:
            rel_path = os.path.relpath(result['file'], self.repo_path)
            output.append(f"📍 {rel_path}:{result['line']}")
            output.append(f"    {result['content']}")
            output.append("")
        
        return "\n".join(output)

    def _get_most_dependent_modules(self, mdg) -> List[Tuple[str, int]]:
        """Get modules with most dependencies"""
        in_degrees = dict(mdg.in_degree())
        return sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]

    def _detect_dependency_cycles(self, mdg) -> List[List[str]]:
        """Detect circular dependencies"""
        try:
            import networkx as nx
            cycles = list(nx.simple_cycles(mdg))
            return cycles[:5]
        except:
            return []

    def _format_dependency_analysis(self, stats: Dict) -> str:
        """Format dependency analysis results"""
        output = ["## Dependency Analysis"]
        output.append(f"Total modules: {stats['total_modules']}")
        output.append(f"Total dependencies: {stats['total_dependencies']}")
        output.append("")
        
        output.append("### Most Dependent Modules:")
        for module, count in stats['most_dependent']:
            rel_path = os.path.relpath(module, self.repo_path) if os.path.isfile(module) else module
            output.append(f"- {rel_path}: {count} dependencies")
        
        if stats['dependency_cycles']:
            output.append("\n### Dependency Cycles Detected:")
            for i, cycle in enumerate(stats['dependency_cycles'], 1):
                output.append(f"Cycle {i}: {' -> '.join(cycle)}")
        
        return "\n".join(output)

    def _find_execution_paths(self, fcg, func_name: str) -> Dict:
        """Find execution paths involving a function"""
        import networkx as nx
        
        paths = {
            'callers': [],
            'callees': [],
            'paths_to': [],
            'paths_from': []
        }
        
        for edge in fcg.edges():
            caller, callee = edge
            if func_name in caller:
                paths['callees'].append(callee)
            if func_name in callee:
                paths['callers'].append(caller)
        
        return paths

    def _format_execution_paths(self, paths: Dict) -> str:
        """Format execution path analysis"""
        output = ["## Execution Path Analysis"]
        
        if paths['callers']:
            output.append("### Functions that call this:")
            for caller in paths['callers'][:5]:
                output.append(f"- {caller}")
        
        if paths['callees']:
            output.append("\n### Functions called by this:")
            for callee in paths['callees'][:5]:
                output.append(f"- {callee}")
        
        return "\n".join(output)

    def _extract_path(self, query: str) -> Optional[str]:
        """Extract file/directory path from query"""
        import re
        
        path_pattern = r'["\']([^"\']+)["\']|(\S+/\S+)'
        match = re.search(path_pattern, query)
        
        if match:
            return match.group(1) or match.group(2)
        
        return None

# ==============================================================================
# Task Execution and Environment Management
# ==============================================================================

class TaskExecutor:
    """Execute tasks in controlled environment"""
    
    def __init__(self, work_dir: str, config: CodeExecutionConfig):
        self.work_dir = work_dir
        self.config = config
        self.venv_path = None
        
        os.makedirs(self.work_dir, exist_ok=True)
    
    async def setup_environment(self, repo_path: str) -> bool:
        """Setup execution environment"""
        try:
            if self.config.use_venv:
                await self._setup_virtualenv()
            
            await self._install_dependencies(repo_path)
            
            return True
        
        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            return False
    
    async def execute_code(self, code: str, file_name: str = "task_script.py") -> Dict[str, Any]:
        """Execute code in the controlled environment"""
        script_path = os.path.join(self.work_dir, file_name)
        
        try:
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            result = await self._run_script(script_path)
            
            return {
                'success': result['returncode'] == 0,
                'stdout': result['stdout'],
                'stderr': result['stderr'],
                'returncode': result['returncode']
            }
        
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'returncode': -1
            }

    async def _setup_virtualenv(self):
        """Setup virtual environment"""
        self.venv_path = os.path.join(self.work_dir, "venv")
        
        if not os.path.exists(self.venv_path):
            result = await asyncio.create_subprocess_exec(
                sys.executable, '-m', 'venv', self.venv_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.communicate()

    async def _install_dependencies(self, repo_path: str):
        """Install repository dependencies"""
        requirements_files = [
            'requirements.txt',
            'requirements-dev.txt',
            'setup.py',
            'pyproject.toml'
        ]
        
        for req_file in requirements_files:
            req_path = os.path.join(repo_path, req_file)
            if os.path.exists(req_path):
                await self._install_from_file(req_path)
                break

    async def _install_from_file(self, req_path: str):
        """Install dependencies from requirements file"""
        python_cmd = self._get_python_command()
        
        if req_path.endswith('requirements.txt'):
            cmd = [python_cmd, '-m', 'pip', 'install', '-r', req_path]
        elif req_path.endswith('setup.py'):
            cmd = [python_cmd, req_path, 'install']
        else:
            return
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.path.dirname(req_path)
            )
            await process.communicate()
        except Exception as e:
            logger.warning(f"Failed to install dependencies: {e}")

    async def _run_script(self, script_path: str) -> Dict[str, Any]:
        """Run Python script"""
        python_cmd = self._get_python_command()
        
        process = await asyncio.create_subprocess_exec(
            python_cmd, script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.work_dir
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=self.config.timeout
            )
            
            return {
                'returncode': process.returncode,
                'stdout': stdout.decode('utf-8', errors='ignore'),
                'stderr': stderr.decode('utf-8', errors='ignore')
            }
        
        except asyncio.TimeoutError:
            process.kill()
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': 'Execution timed out'
            }

    def _get_python_command(self) -> str:
        """Get Python command (considering virtual environment)"""
        if self.venv_path and os.path.exists(self.venv_path):
            if sys.platform == "win32":
                return os.path.join(self.venv_path, "Scripts", "python.exe")
            else:
                return os.path.join(self.venv_path, "bin", "python")
        return sys.executable

# ==============================================================================
# Repository Management
# ==============================================================================

class RepositoryManager:
    """Manage repository cloning and local operations"""
    
    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self.repos_dir = os.path.join(work_dir, "repositories")
        os.makedirs(self.repos_dir, exist_ok=True)
    
    async def clone_repository(self, repo_url: str, repo_name: str = None) -> str:
        """Clone repository from URL"""
        if not repo_name:
            repo_name = repo_url.split('/')[-1].replace('.git', '')
        
        repo_path = os.path.join(self.repos_dir, repo_name)
        
        if os.path.exists(repo_path):
            logger.info(f"Repository already exists: {repo_path}")
            return repo_path
        
        try:
            process = await asyncio.create_subprocess_exec(
                'git', 'clone', repo_url, repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"Successfully cloned repository: {repo_path}")
                return repo_path
            else:
                logger.error(f"Git clone failed: {stderr.decode()}")
                return None
        
        except Exception as e:
            logger.error(f"Repository cloning error: {e}")
            return None

    def get_repository_info(self, repo_path: str) -> Dict[str, Any]:
        """Get repository information"""
        info = {
            'path': repo_path,
            'name': os.path.basename(repo_path),
            'files': [],
            'size': 0
        }
        
        try:
            for root, dirs, files in os.walk(repo_path):
                if '.git' in root:
                    continue
                
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        size = os.path.getsize(file_path)
                        info['files'].append({
                            'path': os.path.relpath(file_path, repo_path),
                            'size': size
                        })
                        info['size'] += size
                    except OSError:
                        continue
            
            info['commits'] = self._get_recent_commits(repo_path)
            
        except Exception as e:
            logger.warning(f"Error getting repository info: {e}")
        
        return info

    def _get_recent_commits(self, repo_path: str, count: int = 5) -> List[Dict]:
        """Get recent commits"""
        commits = []
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'log', f'--max-count={count}', '--pretty=format:%h|%an|%ad|%s'],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        hash_val, author, date, message = line.split('|', 3)
                        commits.append({
                            'hash': hash_val,
                            'author': author,
                            'date': date,
                            'message': message
                        })
        except Exception:
            pass
        
        return commits

# ==============================================================================
# Main RepoMaster Agent
# ==============================================================================

class RepoMasterAgent:
    """Main RepoMaster agent orchestrating the entire workflow"""
    
    def __init__(self,
                 llm_config: Dict,
                 code_execution_config: Dict,
                 explorer_config: Dict = None):
        
        self.llm_config = LLMConfig(**self._extract_llm_config(llm_config))
        self.code_execution_config = CodeExecutionConfig(**code_execution_config)
        self.explorer_config = ExplorerConfig(**(explorer_config or {}))
        
        self.llm = LLMBackend(self.llm_config)
        self.repo_searcher = RepositorySearcher()
        self.repo_manager = RepositoryManager(self.code_execution_config.work_dir)
        self.task_executor = TaskExecutor(
            self.code_execution_config.work_dir,
            self.code_execution_config
        )
        
        self.current_repositories = []
        self.current_task = None
        self.execution_history = []

    def _extract_llm_config(self, config_dict: Dict) -> Dict:
        """Extract LLM config from AutoGen-style config"""
        if 'config_list' in config_dict and config_dict['config_list']:
            llm_config = config_dict['config_list'][0].copy()
            
            for key in ['timeout', 'temperature']:
                if key in config_dict:
                    llm_config[key] = config_dict[key]
            
            return llm_config
        
        return config_dict

    async def solve_task_with_repo(self, task_description: str) -> Dict[str, Any]:
        """Main method to solve task using repository exploration"""
        self.current_task = task_description
        
        try:
            logger.info("🔍 Searching for relevant repositories...")
            repositories = await self.repo_searcher.search_repositories(
                task_description, max_repos=10
            )
            
            if not repositories:
                return {
                    'success': False,
                    'error': 'No relevant repositories found',
                    'repositories': []
                }
            
            self.current_repositories = repositories
            
            logger.info("📥 Cloning and analyzing repositories...")
            analysis_results = []
            
            for repo in repositories[:3]:
                repo_path = await self.repo_manager.clone_repository(repo.url, repo.name.split('/')[-1])
                
                if repo_path:
                    explorer = CodeExplorer(
                        repo_path=repo_path,
                        work_dir=self.code_execution_config.work_dir,
                        llm_config=self.llm_config,
                        explorer_config=self.explorer_config
                    )
                    
                    context = await explorer.initialize_exploration(repo)
                    
                    analysis_results.append({
                        'repository': repo,
                        'path': repo_path,
                        'explorer': explorer,
                        'context': context
                    })
            
            logger.info("🎯 Executing task with selected repository...")
            best_result = await self._execute_task_with_best_repo(
                task_description, analysis_results
            )
            
            return best_result
        
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'task': task_description
            }

    async def _execute_task_with_best_repo(self,
                                             task_description: str,
                                             analysis_results: List[Dict]) -> Dict[str, Any]:
        """Execute task with the best matching repository"""
        
        for result in analysis_results:
            try:
                explorer = result['explorer']
                repo_path = result['path']
                
                env_setup = await self.task_executor.setup_environment(repo_path)
                if not env_setup:
                    continue
                
                execution_plan = await self._generate_execution_plan(
                    task_description, result['context'], explorer
                )
                
                execution_result = await self._execute_plan(execution_plan, explorer)
                
                if execution_result['success']:
                    return {
                        'success': True,
                        'repository': result['repository'].name,
                        'result': execution_result,
                        'plan': execution_plan
                    }
            
            except Exception as e:
                logger.warning(f"Execution failed for {result['repository'].name}: {e}")
                continue
        
        return {
            'success': False,
            'error': 'All repository execution attempts failed'
        }

    async def _generate_execution_plan(self,
                                       task_description: str,
                                       context: str,
                                       explorer: CodeExplorer) -> Dict[str, Any]:
        """Generate execution plan based on task and repository context"""
        
        planning_prompt = f"""
        You are a code execution planner. Based on the task description and repository context,
        generate a detailed execution plan.
        
        Task: {task_description}
        
        Repository Context:
        {context}
        
        Generate a JSON execution plan with the following structure:
        {{
            "steps": [
                {{
                    "action": "explore_code|execute_script|modify_file",
                    "description": "Step description",
                    "details": "Specific implementation details",
                    "files": ["list of files involved"],
                    "code": "code to execute (if applicable)"
                }}
            ],
            "expected_output": "Description of expected results",
            "dependencies": ["list of required dependencies"]
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are an expert code execution planner."},
            {"role": "user", "content": planning_prompt}
        ]
        
        response = await self.llm.generate_response(messages)
        
        try:
            plan = json.loads(response)
            return plan
        except json.JSONDecodeError:
            return {
                "steps": [
                    {
                        "action": "explore_code",
                        "description": "Explore repository structure",
                        "details": "Understand the codebase organization",
                        "files": [],
                        "code": ""
                    }
                ],
                "expected_output": "Repository understanding",
                "dependencies": []
            }

    async def _execute_plan(self, plan: Dict[str, Any], explorer: CodeExplorer) -> Dict[str, Any]:
        """Execute the generated plan"""
        results = []
        
        try:
            for step in plan.get('steps', []):
                action = step.get('action', '')
                
                if action == 'explore_code':
                    result = await self._execute_exploration_step(step, explorer)
                elif action == 'execute_script':
                    result = await self._execute_script_step(step)
                elif action == 'modify_file':
                    result = await self._execute_modification_step(step)
                else:
                    result = {'success': False, 'error': f'Unknown action: {action}'}
                
                results.append(result)
                
                if not result.get('success', False) and step.get('critical', False):
                    break
            
            return {
                'success': all(r.get('success', False) for r in results),
                'steps': results,
                'plan': plan
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'steps': results
            }

    async def _execute_exploration_step(self, step: Dict, explorer: CodeExplorer) -> Dict[str, Any]:
        """Execute code exploration step"""
        try:
            query = step.get('description', '') + ' ' + step.get('details', '')
            result = await explorer.explore_code(query)
            
            return {
                'success': True,
                'action': 'explore_code',
                'result': result
            }
        
        except Exception as e:
            return {
                'success': False,
                'action': 'explore_code',
                'error': str(e)
            }

    async def _execute_script_step(self, step: Dict) -> Dict[str, Any]:
        """Execute script step"""
        try:
            code = step.get('code', '')
            if not code:
                return {'success': False, 'error': 'No code to execute'}
            
            result = await self.task_executor.execute_code(code)
            
            return {
                'success': result['success'],
                'action': 'execute_script',
                'result': result
            }
        
        except Exception as e:
            return {
                'success': False,
                'action': 'execute_script',
                'error': str(e)
            }

    async def _execute_modification_step(self, step: Dict) -> Dict[str, Any]:
        """Execute file modification step"""
        try:
            files = step.get('files', [])
            code = step.get('code', '')
            
            if not files or not code:
                return {'success': False, 'error': 'Missing files or code for modification'}
            
            for file_path in files:
                full_path = os.path.join(self.code_execution_config.work_dir, file_path)
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(code)
            
            return {
                'success': True,
                'action': 'modify_file',
                'files': files
            }
        
        except Exception as e:
            return {
                'success': False,
                'action': 'modify_file',
                'error': str(e)
            }

# ==============================================================================
# Usage Example and Main Entry Point
# ==============================================================================

# Continuation from the main function and additional components

async def main():
    """Example usage of RepoMaster"""
    # Configuration
    llm_config = {
        'config_list': [{
            'model': 'claude-3-5-sonnet-20241022',
            'api_key': os.getenv('ANTHROPIC_API_KEY'),
            'temperature': 0.1,
            'max_tokens': 4000,
            'timeout': 2000
        }]
    }
    
    code_execution_config = {
        'work_dir': 'workspace',
        'use_docker': False,
        'timeout': 7200,
        'use_venv': True
    }
    
    explorer_config = {
        'max_turns': 40,
        'function_call': True,
        'repo_init': True,
        'max_context_length': 8000,
        'compression_ratio': 0.3
    }

    # Initialize RepoMaster
    repo_master = RepoMasterAgent(
        llm_config=llm_config,
        code_execution_config=code_execution_config,
        explorer_config=explorer_config,
        github_token=os.getenv('GITHUB_TOKEN')
    )

    # Example task - you can modify this based on your needs
    task = """
    I need to implement a web scraper that can handle JavaScript-rendered pages. 
    Find a repository that demonstrates how to use Selenium or similar tools for 
    web scraping, and help me adapt it to scrape product information from an 
    e-commerce website including prices, descriptions, and availability status.
    """

    try:
        # Execute the task
        print("Starting RepoMaster task execution...")
        result = await repo_master.solve_task_with_repo(task)
        
        # Display results
        print("\n" + "="*80)
        print("TASK EXECUTION RESULTS")
        print("="*80)
        
        if result.get('success'):
            print(f"✅ Task completed successfully!")
            print(f"Repository used: {result.get('repository_name', 'Unknown')}")
            print(f"Execution time: {result.get('execution_time', 'Unknown')} seconds")
            
            if result.get('generated_code'):
                print(f"\n📝 Generated Code:")
                print("-" * 40)
                print(result['generated_code'])
            
            if result.get('execution_output'):
                print(f"\n🏃 Execution Output:")
                print("-" * 40)
                print(result['execution_output'])
                
            if result.get('files_modified'):
                print(f"\n📁 Modified Files:")
                for file_path in result['files_modified']:
                    print(f"  - {file_path}")
                    
        else:
            print(f"❌ Task failed: {result.get('error', 'Unknown error')}")
            
        # Display token usage statistics
        if result.get('token_stats'):
            stats = result['token_stats']
            print(f"\n📊 Token Usage Statistics:")
            print(f"  Total tokens used: {stats.get('total_tokens', 0):,}")
            print(f"  Average tokens per step: {stats.get('avg_tokens_per_step', 0):,}")
            print(f"  Context compression ratio: {stats.get('compression_ratio', 0):.2%}")

    except Exception as e:
        print(f"❌ Error during task execution: {e}")
        import traceback
        traceback.print_exc()

# Additional Helper Classes and Methods

class TokenUsageTracker:
    """Track token usage throughout the exploration process"""
    
    def __init__(self):
        self.total_tokens = 0
        self.step_tokens = []
        self.context_sizes = []
        
    def add_usage(self, tokens: int, context_size: int = 0):
        """Add token usage for a step"""
        self.total_tokens += tokens
        self.step_tokens.append(tokens)
        if context_size > 0:
            self.context_sizes.append(context_size)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            'total_tokens': self.total_tokens,
            'avg_tokens_per_step': sum(self.step_tokens) / len(self.step_tokens) if self.step_tokens else 0,
            'max_tokens_step': max(self.step_tokens) if self.step_tokens else 0,
            'min_tokens_step': min(self.step_tokens) if self.step_tokens else 0,
            'avg_context_size': sum(self.context_sizes) / len(self.context_sizes) if self.context_sizes else 0,
            'compression_ratio': 1 - (sum(self.context_sizes) / (self.total_tokens * 4)) if self.total_tokens > 0 else 0
        }

# Enhanced RepoMasterAgent with missing methods
class EnhancedRepoMasterAgent(RepoMasterAgent):
    """Enhanced RepoMaster with additional features from the paper"""
    
    def __init__(self, llm_config: Dict, code_execution_config: Dict, explorer_config: Dict, github_token: str = None):
        super().__init__(llm_config, code_execution_config, explorer_config, github_token)
        self.token_tracker = TokenUsageTracker()
        self.execution_history = []
        
    async def solve_task_with_repo(self, task_description: str) -> Dict[str, Any]:
        """Enhanced task solving with detailed tracking and error recovery"""
        start_time = time.time()
        
        try:
            # Step 1: Repository Discovery
            print("🔍 Searching for relevant repositories...")
            repositories = await self.repo_searcher.search_repositories(
                task_description, max_repos=self.explorer_config.max_repos
            )
            
            if not repositories:
                return {
                    'success': False,
                    'error': 'No relevant repositories found',
                    'execution_time': time.time() - start_time
                }
            
            print(f"Found {len(repositories)} candidate repositories")
            
            # Step 2: Repository Analysis and Ranking
            analysis_results = []
            for i, repo in enumerate(repositories[:3]):  # Analyze top 3
                print(f"📊 Analyzing repository {i+1}/{min(3, len(repositories))}: {repo.name}")
                
                try:
                    # Clone repository
                    repo_path = await self.repo_manager.clone_repository(repo.url, repo.name.split('/')[-1])
                    
                    # Analyze structure
                    analyzer = CodeStructureAnalyzer()
                    analysis = analyzer.analyze_repository(repo_path)
                    
                    # Initialize explorer
                    explorer = CodeExplorer(
                        repo_path=repo_path,
                        llm=self.llm,
                        config=self.explorer_config,
                        analysis_result=analysis
                    )
                    
                    # Get initial context
                    context = await explorer.initialize_exploration(repo)
                    
                    # Calculate relevance score
                    relevance_score = await self._calculate_repo_relevance(repo, task_description, context)
                    
                    analysis_results.append({
                        'repository': repo,
                        'path': repo_path,
                        'analysis': analysis,
                        'explorer': explorer,
                        'context': context,
                        'relevance_score': relevance_score
                    })
                    
                except Exception as e:
                    print(f"⚠️ Failed to analyze {repo.name}: {e}")
                    continue
            
            if not analysis_results:
                return {
                    'success': False,
                    'error': 'Failed to analyze any repositories',
                    'execution_time': time.time() - start_time
                }
            
            # Step 3: Select best repository
            best_result = max(analysis_results, key=lambda x: x['relevance_score'])
            print(f"🎯 Selected repository: {best_result['repository'].name} (score: {best_result['relevance_score']:.3f})")
            
            # Step 4: Execute task with best repository
            execution_result = await self._execute_task_with_best_repo(
                task_description, best_result
            )
            
            # Step 5: Compile final results
            end_time = time.time()
            
            return {
                'success': execution_result.get('success', False),
                'repository_name': best_result['repository'].name,
                'repository_url': best_result['repository'].url,
                'generated_code': execution_result.get('generated_code'),
                'execution_output': execution_result.get('execution_output'),
                'files_modified': execution_result.get('files_modified', []),
                'execution_time': end_time - start_time,
                'token_stats': self.token_tracker.get_stats(),
                'error': execution_result.get('error') if not execution_result.get('success') else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time,
                'token_stats': self.token_tracker.get_stats()
            }
    
    async def _calculate_repo_relevance(self, repo: RepositoryInfo, task: str, context: str) -> float:
        """Calculate repository relevance score using LLM"""
        
        relevance_prompt = f"""
        Analyze how relevant this repository is for solving the given task.
        
        Task: {task}
        
        Repository Information:
        - Name: {repo.name}
        - Description: {repo.description}
        - Stars: {repo.stars}
        - Language: {repo.language}
        - Topics: {', '.join(repo.topics)}
        
        Repository Context:
        {context[:2000]}...
        
        Rate the relevance on a scale of 0.0 to 1.0, where:
        - 0.0-0.3: Not relevant or minimal relevance
        - 0.3-0.6: Somewhat relevant, might be useful
        - 0.6-0.8: Highly relevant, good match
        - 0.8-1.0: Perfect match, exactly what's needed
        
        Consider:
        1. How well the repository's purpose matches the task
        2. Code quality and structure
        3. Documentation and examples
        4. Maintenance and popularity
        
        Respond with only a single number between 0.0 and 1.0.
        """
        
        try:
            messages = [{'role': 'user', 'content': relevance_prompt}]
            response = await self.llm.generate_response(messages)
            score = float(response.strip())
            self.token_tracker.add_usage(len(relevance_prompt.split()) * 1.3)  # Rough token estimate
            return max(0.0, min(1.0, score))  # Clamp to valid range
        except:
            return 0.5  # Default score if evaluation fails
    
    async def _execute_task_with_best_repo(self, task_description: str, result: Dict) -> Dict[str, Any]:
        """Execute task with enhanced error handling and multi-step approach"""
        
        explorer = result['explorer']
        repo_path = result['path']
        
        try:
            # Setup execution environment
            print("🔧 Setting up execution environment...")
            env_setup = await self.task_executor.setup_environment(repo_path)
            if not env_setup:
                print("⚠️ Environment setup had issues, continuing anyway...")
            
            # Generate comprehensive execution plan
            print("📋 Generating execution plan...")
            execution_plan = await self._generate_comprehensive_execution_plan(
                task_description, result['context'], result['analysis']
            )
            
            # Execute plan with error recovery
            print("🚀 Executing plan...")
            execution_result = await self._execute_plan_with_recovery(execution_plan, explorer)
            
            return execution_result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Task execution failed: {str(e)}',
                'generated_code': None,
                'execution_output': None
            }
    
    async def _generate_comprehensive_execution_plan(self, task: str, context: str, analysis: Dict) -> Dict[str, Any]:
        """Generate a more detailed execution plan"""
        
        # Get core components for better context
        core_components = []
        if 'hct' in analysis:
            analyzer = CodeStructureAnalyzer()
            analyzer.hct = analysis['hct']
            analyzer.fcg = analysis.get('fcg', nx.DiGraph())
            analyzer.mdg = analysis.get('mdg', nx.DiGraph())
            analyzer._calculate_importance_scores()
            core_components = analyzer.get_core_components(10)
        
        planning_prompt = f"""
        Create a detailed execution plan to solve this task using the provided repository.
        
        Task: {task}
        
        Repository Context:
        {context}
        
        Core Components:
        {chr(10).join([f"- {comp[0]} (importance: {comp[1]:.3f})" for comp in core_components[:5]])}
        
        Create a JSON plan with the following structure:
        {{
            "overview": "Brief description of the approach",
            "steps": [
                {{
                    "id": 1,
                    "action": "explore|script|modify",
                    "description": "What this step does",
                    "details": "Specific details or code",
                    "expected_output": "What should happen",
                    "error_recovery": "What to do if this fails"
                }}
            ],
            "success_criteria": "How to determine if the task is complete",
            "deliverables": ["List of expected outputs/files"]
        }}
        
        Action types:
        - "explore": Examine code, understand structure, find relevant components
        - "script": Write and execute code to solve the task
        - "modify": Modify existing files in the repository
        
        Make the plan comprehensive but efficient. Focus on understanding the repository
        structure first, then implementing the solution.
        """
        
        try:
            messages = [{'role': 'user', 'content': planning_prompt}]
            response = await self.llm.generate_response(messages)
            self.token_tracker.add_usage(len(planning_prompt.split()) * 1.3)
            
            # Clean response and parse JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
                
            plan = json.loads(response)
            return plan
            
        except Exception as e:
            print(f"⚠️ Failed to generate detailed plan: {e}")
            # Fallback to simple plan
            return {
                "overview": "Simple exploration and implementation approach",
                "steps": [
                    {
                        "id": 1,
                        "action": "explore",
                        "description": "Explore repository structure and find relevant code",
                        "details": "Look for main modules and entry points",
                        "expected_output": "Understanding of codebase structure",
                        "error_recovery": "Continue with basic file listing"
                    },
                    {
                        "id": 2,
                        "action": "script",
                        "description": "Implement solution based on repository patterns",
                        "details": "Write code that adapts existing patterns for the task",
                        "expected_output": "Working implementation",
                        "error_recovery": "Simplify approach and retry"
                    }
                ],
                "success_criteria": "Code runs without errors and produces expected output",
                "deliverables": ["Working script", "Documentation"]
            }
    
    async def _execute_plan_with_recovery(self, plan: Dict[str, Any], explorer: CodeExplorer) -> Dict[str, Any]:
        """Execute plan with error recovery and detailed tracking"""
        
        execution_log = []
        generated_files = []
        all_generated_code = []
        
        print(f"📝 Plan Overview: {plan.get('overview', 'No overview provided')}")
        
        for step in plan.get('steps', []):
            step_id = step.get('id', 0)
            action = step.get('action', '')
            description = step.get('description', '')
            
            print(f"  Step {step_id}: {description}")
            
            try:
                if action == 'explore':
                    result = await self._execute_exploration_step(step, explorer)
                elif action == 'script':
                    result = await self._execute_script_step(step)
                    if result.get('success') and result.get('code'):
                        all_generated_code.append(result['code'])
                elif action == 'modify':
                    result = await self._execute_modification_step(step)
                    if result.get('success') and result.get('files'):
                        generated_files.extend(result['files'])
                else:
                    result = {'success': False, 'error': f'Unknown action: {action}'}
                
                execution_log.append({
                    'step_id': step_id,
                    'action': action,
                    'description': description,
                    'success': result.get('success', False),
                    'output': result.get('output', ''),
                    'error': result.get('error', '')
                })
                
                if not result.get('success', False):
                    print(f"    ⚠️ Step failed: {result.get('error', 'Unknown error')}")
                    # Try error recovery if specified
                    recovery = step.get('error_recovery', '')
                    if recovery:
                        print(f"    🔄 Attempting recovery: {recovery}")
                        # Implement basic recovery logic here
                
            except Exception as e:
                print(f"    ❌ Step exception: {e}")
                execution_log.append({
                    'step_id': step_id,
                    'action': action,
                    'description': description,
                    'success': False,
                    'error': str(e)
                })
        
        # Determine overall success
        successful_steps = sum(1 for log in execution_log if log['success'])
        total_steps = len(execution_log)
        success_rate = successful_steps / total_steps if total_steps > 0 else 0
        
        overall_success = success_rate >= 0.7  # At least 70% steps successful
        
        return {
            'success': overall_success,
            'execution_log': execution_log,
            'success_rate': success_rate,
            'generated_code': '\n\n'.join(all_generated_code) if all_generated_code else None,
            'files_modified': generated_files,
            'execution_output': f"Completed {successful_steps}/{total_steps} steps successfully"
        }

# Enhanced execution methods with better error handling
    async def _execute_exploration_step(self, step: Dict, explorer: CodeExplorer) -> Dict[str, Any]:
        """Execute code exploration step with enhanced context"""
        try:
            query = step.get('description', '') + ' ' + step.get('details', '')
            result = await explorer.explore_code(query)
            
            return {
                'success': True,
                'output': result,
                'exploration_data': result
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Exploration failed: {str(e)}',
                'output': ''
            }

    async def _execute_script_step(self, step: Dict) -> Dict[str, Any]:
        """Execute script step with code generation if needed"""
        try:
            code = step.get('details', '')
            
            # If no code provided, generate it based on description
            if not code or len(code.strip()) < 20:
                code = await self._generate_code_for_step(step)
            
            if code:
                result = await self.task_executor.execute_code(code)
                return {
                    'success': result.get('success', False),
                    'output': result.get('output', ''),
                    'error': result.get('stderr', ''),
                    'code': code
                }
            else:
                return {
                    'success': False,
                    'error': 'No code to execute',
                    'output': ''
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Script execution failed: {str(e)}',
                'output': ''
            }

    async def _generate_code_for_step(self, step: Dict) -> str:
        """Generate code for a step based on its description"""
        code_prompt = f"""
        Generate Python code to accomplish this task step:
        
        Description: {step.get('description', '')}
        Details: {step.get('details', '')}
        Expected Output: {step.get('expected_output', '')}
        
        Write clean, working Python code that accomplishes this step.
        Include error handling and comments.
        Only return the code, no explanations.
        """
        
        try:
            messages = [{'role': 'user', 'content': code_prompt}]
            response = await self.llm.generate_response(messages)
            self.token_tracker.add_usage(len(code_prompt.split()) * 1.3)
            
            # Clean up the response
            code = response.strip()
            if code.startswith('```python'):
                code = code[9:]
            elif code.startswith('```'):
                code = code[3:]
            if code.endswith('```'):
                code = code[:-3]
                
            return code.strip()
            
        except Exception as e:
            print(f"⚠️ Code generation failed: {e}")
            return ""

# CLI Interface
def create_cli():
    """Create command-line interface for RepoMaster"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RepoMaster: Autonomous Repository Exploration")
    parser.add_argument("task", help="Task description to solve")
    parser.add_argument("--model", default="claude-3-5-sonnet-20241022", help="LLM model to use")
    parser.add_argument("--max-repos", type=int, default=5, help="Maximum repositories to analyze")
    parser.add_argument("--work-dir", default="workspace", help="Working directory")
    parser.add_argument("--timeout", type=int, default=7200, help="Execution timeout in seconds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    return parser

async def cli_main():
    """CLI entry point"""
    parser = create_cli()
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Configuration
    llm_config = {
        'config_list': [{
            'model': args.model,
            'api_key': os.getenv('ANTHROPIC_API_KEY') or os.getenv('OPENAI_API_KEY'),
            'temperature': 0.1,
            'max_tokens': 4000,
            'timeout': 2000
        }]
    }
    
    code_execution_config = {
        'work_dir': args.work_dir,
        'use_docker': False,
        'timeout': args.timeout,
        'use_venv': True
    }
    
    explorer_config = {
        'max_turns': 40,
        'function_call': True,
        'repo_init': True,
        'max_context_length': 8000,
        'compression_ratio': 0.3,
        'max_repos': args.max_repos
    }

    # Initialize and run
    repo_master = EnhancedRepoMasterAgent(
        llm_config=llm_config,
        code_execution_config=code_execution_config,
        explorer_config=explorer_config,
        github_token=os.getenv('GITHUB_TOKEN')
    )

    result = await repo_master.solve_task_with_repo(args.task)
    
    # Output results
    if result.get('success'):
        print(f"\n✅ Task completed successfully!")
        print(f"Repository: {result.get('repository_name')}")
        if result.get('generated_code'):
            print(f"\nGenerated Code:\n{'-'*40}")
            print(result['generated_code'])
    else:
        print(f"\n❌ Task failed: {result.get('error')}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # CLI mode
        asyncio.run(cli_main())
    else:
        # Interactive mode
        asyncio.run(main())
