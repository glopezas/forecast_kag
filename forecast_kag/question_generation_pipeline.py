"""
Binary Forecasting Question Generation Pipeline
Generates high-quality binary forecasting questions by "time traveling" - finding resolved past events
and framing them as future uncertainties.

Pipeline consists of three modules:
1. The Architect (Taxonomy Expander): Breaks broad topic into granular subcategories
2. The Historian (Generator): Finds resolved events and backdates them to create questions
3. The Critic (Validator): Independently verifies questions are unambiguous and answers are correct
"""

import yaml
import logging
import json
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
import re
from pathlib import Path

from openai import OpenAI
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class ModelConfig:
    """Load and manage model configurations from YAML file."""

    def __init__(self, config_path: str = "models/model_servers.yaml"):
        self.config_path = config_path
        self.servers = self._load_config()

    def _load_config(self) -> List[Dict[str, str]]:
        """Load server configurations from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('servers', [])

    def get_server_by_shortname(self, shortname: str) -> Optional[Dict[str, str]]:
        """Get server configuration by shortname."""
        for server in self.servers:
            if server['shortname'] == shortname:
                return server
        return None

    def get_all_shortnames(self) -> List[str]:
        """Get list of all available server shortnames."""
        return [server['shortname'] for server in self.servers]


class LLMClient:
    """Client for interacting with OpenAI-compatible LLM endpoints."""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_base = api_base

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        log_context: str = None
    ) -> str:
        """Generate text using the LLM."""
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        if log_context:
            logging.info(f"Calling LLM for {log_context}...")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=tokens
            )

            if log_context:
                logging.info(f"{log_context} completed")

            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)
            logging.error(f"LLM API Error: {error_msg[:200]}")
            raise RuntimeError(
                f"Failed to get response from LLM server at {self.api_base}. "
                f"Error: {error_msg[:200]}. "
                f"Please check that your server is running and accessible."
            ) from e


class SearchEngine:
    """Free search engine interface using DuckDuckGo HTML scraping."""

    def __init__(self, rate_limit_delay: float = 2.0):
        """
        Initialize search engine.

        Args:
            rate_limit_delay: Delay in seconds between requests
        """
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def search(self, query: str, num_results: int = 10, date_range: Optional[Tuple[str, str]] = None) -> List[Dict[str, str]]:
        """
        Search using DuckDuckGo HTML interface.

        Args:
            query: Search query
            num_results: Number of results to return
            date_range: Optional tuple of (start_date, end_date) in "YYYY-MM-DD" format

        Returns:
            List of search result dictionaries with 'title', 'url', 'snippet'
        """
        try:
            # Add date range to query if provided
            if date_range:
                start_date, end_date = date_range
                query = f"{query} after:{start_date} before:{end_date}"

            # DuckDuckGo HTML search
            url = "https://html.duckduckgo.com/html/"
            data = {'q': query}

            time.sleep(self.rate_limit_delay)

            response = self.session.post(url, data=data, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            results = []

            # Parse search results
            for result_div in soup.find_all('div', class_='result')[:num_results]:
                try:
                    title_elem = result_div.find('a', class_='result__a')
                    snippet_elem = result_div.find('a', class_='result__snippet')

                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        url = title_elem.get('href', '')
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''

                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet
                        })
                except Exception as e:
                    logging.debug(f"Error parsing individual result: {e}")
                    continue

            logging.info(f"Search found {len(results)} results for query: '{query[:60]}...'")
            return results

        except Exception as e:
            logging.error(f"Search error for query '{query}': {e}")
            return []

    def format_results_for_llm(self, results: List[Dict[str, str]]) -> str:
        """Format search results for LLM consumption."""
        if not results:
            return "No search results found."

        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"Result {i}:\n"
                f"Title: {result['title']}\n"
                f"URL: {result['url']}\n"
                f"Snippet: {result['snippet']}\n"
            )

        return "\n---\n".join(formatted)


class TaxonomyExpander:
    """Module A: The Architect - Expands a topic into granular subcategories."""

    def __init__(self, llm_client: LLMClient, prompt_path: str = "forecast_kag/prompts/taxonomy_expansion.txt"):
        self.llm_client = llm_client
        with open(prompt_path, 'r') as f:
            self.prompt_template = f.read()

    def expand(
        self,
        topic: str,
        num_categories: int = 5,
        num_subcategories_per_category: int = 4
    ) -> Dict[str, Any]:
        """
        Expand a topic into categories and subcategories.

        Args:
            topic: Broad topic (e.g., "Sports", "Politics")
            num_categories: Number of main categories to generate
            num_subcategories_per_category: Number of subcategories per category

        Returns:
            Dictionary with taxonomy structure and complexity scores
        """
        logging.info("="*80)
        logging.info("[MODULE A: THE ARCHITECT] Taxonomy Expansion")
        logging.info("="*80)
        logging.info(f"Topic: {topic}")
        logging.info(f"Generating {num_categories} categories with {num_subcategories_per_category} subcategories each")

        prompt = self.prompt_template.format(
            topic=topic,
            num_categories=num_categories,
            num_subcategories_per_category=num_subcategories_per_category
        )

        response = self.llm_client.generate(
            prompt,
            temperature=0.7,
            log_context="Taxonomy Expansion"
        )

        # Extract JSON from response
        try:
            # Check if response is valid
            if response is None or not response.strip():
                raise ValueError("LLM returned empty response")

            # Try to find JSON in the response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                taxonomy = json.loads(json_match.group())
            else:
                taxonomy = json.loads(response)

            logging.info(f"Generated {len(taxonomy['categories'])} categories")

            # Calculate total complexity and log categories
            total_complexity = 0
            for cat in taxonomy['categories']:
                complexity = cat['complexity_score']
                total_complexity += complexity
                logging.info(f"  - {cat['name']}: complexity={complexity}, subcategories={len(cat['subcategories'])}")

            taxonomy['total_complexity'] = total_complexity

            return taxonomy

        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Failed to parse taxonomy JSON: {e}")
            if response:
                logging.error(f"Raw response: {response[:500]}")
            else:
                logging.error("Raw response: None")
            raise


class QuestionHistorian:
    """Module B: The Historian - Generates questions first, then searches for answers."""

    def __init__(
        self,
        llm_client: LLMClient,
        search_engine: SearchEngine,
        buffer_days: int = 7,
        min_resolution_buffer_days: int = 30,
        question_prompt_path: str = "forecast_kag/prompts/historian_question_generation.txt",
        answer_prompt_path: str = "forecast_kag/prompts/historian_answer_search.txt"
    ):
        self.llm_client = llm_client
        self.search_engine = search_engine
        self.buffer_days = buffer_days
        self.min_resolution_buffer_days = min_resolution_buffer_days

        # Context tracking: store recent questions per subcategory to avoid repetition
        self.subcategory_context = {}  # {subcategory: [list of recent questions]}
        self.max_context_size = 10  # Keep last N questions for context

        # Load both prompt templates
        with open(question_prompt_path, 'r') as f:
            self.question_prompt_template = f.read()
        with open(answer_prompt_path, 'r') as f:
            self.answer_prompt_template = f.read()

    def _get_context(self, subcategory: str) -> str:
        """Get context of recent questions for this subcategory."""
        if subcategory not in self.subcategory_context:
            return ""

        recent_questions = self.subcategory_context[subcategory]
        if not recent_questions:
            return ""

        context = "\n\nPREVIOUSLY GENERATED QUESTIONS FOR THIS SUBCATEGORY:\n"
        context += "DO NOT generate questions similar to these. Choose DIFFERENT entities, events, and timeframes:\n"
        for i, q in enumerate(recent_questions, 1):
            context += f"{i}. {q}\n"
        context += "\nGenerate a UNIQUE question that is DIFFERENT from all of the above.\n"

        return context

    def _add_to_context(self, subcategory: str, question: str):
        """Add a question to the context for this subcategory."""
        if subcategory not in self.subcategory_context:
            self.subcategory_context[subcategory] = []

        self.subcategory_context[subcategory].append(question)

        # Keep only the most recent N questions
        if len(self.subcategory_context[subcategory]) > self.max_context_size:
            self.subcategory_context[subcategory] = self.subcategory_context[subcategory][-self.max_context_size:]

    def generate_question(
        self,
        topic: str,
        category: str,
        subcategory: str,
        simulated_date: datetime,
        current_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a binary question using two-stage approach:
        1. Generate question without knowing outcome
        2. Search for answer independently

        Args:
            topic: Main topic
            category: Category name
            subcategory: Specific subcategory
            simulated_date: The "present" we're simulating
            current_date: Actual current date

        Returns:
            Question dictionary or None if generation fails
        """
        # STAGE 1: Generate question WITHOUT searching for answer
        logging.debug(f"Stage 1: Generating question for {subcategory}")

        # Get context of recent questions for this subcategory
        context = self._get_context(subcategory)

        question_prompt = self.question_prompt_template.format(
            current_date=current_date.strftime("%Y-%m-%d"),
            simulated_date=simulated_date.strftime("%Y-%m-%d"),
            topic=topic,
            category=category,
            subcategory=subcategory
        )

        # Append context to prompt to avoid repetition
        question_prompt += context

        question_response = self.llm_client.generate(
            question_prompt,
            temperature=1.2,  # High temperature for maximum creativity and diversity
            max_tokens=800,
            log_context=None
        )

        # Parse question generation response
        try:
            if question_response is None or not question_response.strip():
                logging.warning("LLM returned empty response for question generation")
                return None

            json_match = re.search(r'\{[\s\S]*\}', question_response)
            if json_match:
                question_data = json.loads(json_match.group())
            else:
                question_data = json.loads(question_response)

            question = question_data.get('question')
            entity = question_data.get('entity')
            deadline = question_data.get('deadline')

            if not question or not deadline:
                logging.warning("Question generation missing required fields")
                return None

            # Validate deadline is in the past (so we can verify)
            deadline_date = datetime.strptime(deadline, "%Y-%m-%d")
            if deadline_date > current_date:
                logging.warning(f"Deadline {deadline} is in the future, cannot verify yet")
                return None

            # CRITICAL: Validate deadline has passed with enough buffer for information to be available
            # Require at least min_resolution_buffer_days between deadline and current_date
            days_since_deadline = (current_date - deadline_date).days
            if days_since_deadline < self.min_resolution_buffer_days:
                logging.warning(
                    f"Deadline {deadline} is too recent ({days_since_deadline} days ago). "
                    f"Need at least {self.min_resolution_buffer_days} days for information to be publicly available."
                )
                return None

            # Validate deadline is after simulated date
            if deadline_date < simulated_date:
                logging.warning(f"Deadline {deadline} is before simulated date")
                return None

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logging.error(f"Failed to parse question generation response: {e}")
            logging.debug(f"Raw response: {question_response[:500]}")
            return None

        # STAGE 2: Search for answer independently
        logging.debug(f"Stage 2: Searching for answer to: {question}")

        # Create search query from the question
        search_query = f"{entity} {category} {deadline_date.year}"
        search_results = self.search_engine.search(
            search_query,
            num_results=10
        )

        if not search_results:
            logging.warning(f"No search results to verify answer for: {question}")
            return None

        formatted_results = self.search_engine.format_results_for_llm(search_results)

        # Search for answer using second prompt
        answer_prompt = self.answer_prompt_template.format(
            question=question,
            deadline=deadline,
            entity=entity,
            search_results=formatted_results
        )

        answer_response = self.llm_client.generate(
            answer_prompt,
            temperature=0.3,  # Lower temperature for factual accuracy
            max_tokens=800,
            log_context=None
        )

        # Parse answer search response
        try:
            if answer_response is None or not answer_response.strip():
                logging.warning("LLM returned empty response for answer search")
                return None

            json_match = re.search(r'\{[\s\S]*\}', answer_response)
            if json_match:
                answer_data = json.loads(json_match.group())
            else:
                answer_data = json.loads(answer_response)

            answer = answer_data.get('answer')
            actual_outcome = answer_data.get('actual_outcome')
            actual_date = answer_data.get('actual_date')

            if not answer:
                logging.warning("Answer search missing required fields")
                return None

            # Calculate t_ask and t_resolve
            if actual_date:
                t_resolve = datetime.strptime(actual_date, "%Y-%m-%d")
                t_ask = t_resolve - timedelta(days=self.buffer_days)

                # Validate t_ask is after simulated_date
                if t_ask < simulated_date:
                    logging.warning(f"Question date {t_ask} is before simulated date")
                    return None
            else:
                # Event didn't happen - use deadline as resolution date
                t_resolve = deadline_date
                t_ask = deadline_date - timedelta(days=self.buffer_days)

            # Combine question and answer data
            result = {
                'question': question,
                'answer': answer,
                't_ask': t_ask.strftime("%Y-%m-%d"),
                't_resolve': t_resolve.strftime("%Y-%m-%d"),
                'event_description': actual_outcome,
                'confidence': answer_data.get('confidence', 'Medium'),
                'topic': topic,
                'category': category,
                'subcategory': subcategory
            }

            # Add this question to the context to avoid repeating it
            self._add_to_context(subcategory, question)

            return result

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logging.error(f"Failed to parse answer search response: {e}")
            logging.debug(f"Raw response: {answer_response[:500]}")
            return None


class QuestionCritic:
    """Module C: The Critic - Independently validates questions."""

    def __init__(
        self,
        llm_client: LLMClient,
        search_engine: SearchEngine,
        prompt_path: str = "forecast_kag/prompts/critic_validation.txt"
    ):
        self.llm_client = llm_client
        self.search_engine = search_engine
        with open(prompt_path, 'r') as f:
            self.prompt_template = f.read()

    def validate_question(
        self,
        question_data: Dict[str, Any],
        current_date: datetime
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Independently validate a question.

        Args:
            question_data: Question dictionary from Historian
            current_date: Actual current date

        Returns:
            Tuple of (is_valid, validation_details)
        """
        question = question_data['question']
        t_ask = question_data['t_ask']

        # Perform independent search
        search_results = self.search_engine.search(
            question,
            num_results=10
        )

        formatted_results = self.search_engine.format_results_for_llm(search_results)

        # Validate using LLM
        prompt = self.prompt_template.format(
            question=question,
            t_ask=t_ask,
            current_date=current_date.strftime("%Y-%m-%d"),
            search_results=formatted_results
        )

        response = self.llm_client.generate(
            prompt,
            temperature=0.3,  # Lower temperature for strict validation
            max_tokens=1000,
            log_context=None
        )

        # Parse validation response
        try:
            # Check if response is valid
            if response is None or not response.strip():
                logging.warning("LLM returned empty response for validation")
                return False, {'error': 'Empty LLM response'}

            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                validation = json.loads(json_match.group())
            else:
                validation = json.loads(response)

            # Check if validation passed
            is_valid = validation.get('overall_validation') == 'Pass'

            # Check if answers match
            critic_answer = validation.get('answer', '').strip().lower()
            historian_answer = question_data.get('answer', '').strip().lower()

            answers_match = critic_answer == historian_answer

            if not answers_match:
                logging.warning(
                    f"Answer mismatch - Historian: {historian_answer}, Critic: {critic_answer}"
                )
                is_valid = False
                validation['answer_mismatch'] = True

            return is_valid, validation

        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Failed to parse critic response: {e}")
            logging.debug(f"Raw response: {response[:500]}")
            return False, {'error': str(e)}


class CheckpointManager:
    """Manages checkpointing and recovery for crash-proof operation."""

    def __init__(self, checkpoint_file: str = "completed_log.txt"):
        self.checkpoint_file = checkpoint_file
        self.completed = self._load_checkpoint()

    def _load_checkpoint(self) -> set:
        """Load completed subcategories from checkpoint file."""
        if not os.path.exists(self.checkpoint_file):
            return set()

        with open(self.checkpoint_file, 'r') as f:
            return set(line.strip() for line in f if line.strip())

    def is_completed(self, topic: str, category: str, subcategory: str) -> bool:
        """Check if a subcategory has been processed."""
        key = f"{topic}|{category}|{subcategory}"
        return key in self.completed

    def mark_completed(self, topic: str, category: str, subcategory: str):
        """Mark a subcategory as completed."""
        key = f"{topic}|{category}|{subcategory}"
        self.completed.add(key)

        # Append to file and flush
        with open(self.checkpoint_file, 'a') as f:
            f.write(f"{key}\n")
            f.flush()
            os.fsync(f.fileno())


class QuestionDatabase:
    """Manages JSONL database of generated questions."""

    def __init__(
        self,
        output_file: str = "generated_questions.jsonl",
        discarded_file: str = "discarded_questions.jsonl"
    ):
        self.output_file = output_file
        self.discarded_file = discarded_file
        self._loaded_questions = None  # Cache for duplicate checking

    def _normalize_question(self, question: str) -> str:
        """Normalize question text for duplicate detection."""
        # Remove punctuation, lowercase, remove extra spaces
        import string
        normalized = question.lower()
        normalized = normalized.translate(str.maketrans('', '', string.punctuation))
        normalized = ' '.join(normalized.split())
        return normalized

    def _load_existing_questions(self) -> set:
        """Load existing question texts for duplicate checking."""
        if self._loaded_questions is not None:
            return self._loaded_questions

        questions = set()
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        normalized = self._normalize_question(data['question'])
                        questions.add(normalized)
            except Exception as e:
                logging.warning(f"Failed to load existing questions: {e}")

        self._loaded_questions = questions
        return questions

    def is_duplicate(self, question: str) -> bool:
        """Check if question already exists (case-insensitive, punctuation-insensitive)."""
        existing = self._load_existing_questions()
        normalized = self._normalize_question(question)
        return normalized in existing

    def save_question(self, question_data: Dict[str, Any], validation_data: Dict[str, Any]):
        """Save a validated question to the database."""
        # Create final record
        record = {
            'id': str(uuid.uuid4()),
            'topic': question_data['topic'],
            'category': question_data['category'],
            'subcategory': question_data['subcategory'],
            'question': question_data['question'],
            't_ask': question_data['t_ask'],
            't_resolve': question_data['t_resolve'],
            'answer': question_data['answer'],
            'meta': {
                'complexity_score': question_data.get('complexity_score'),
                'validation_confidence': validation_data.get('confidence', 'Unknown'),
                'event_description': question_data.get('event_description', '')
            }
        }

        # Append to JSONL file and flush
        with open(self.output_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
            f.flush()
            os.fsync(f.fileno())

        # Update cache to prevent duplicates in the same session
        if self._loaded_questions is not None:
            normalized = self._normalize_question(question_data['question'])
            self._loaded_questions.add(normalized)

        logging.info(f"✓ Saved question: {question_data['question'][:80]}...")

    def save_discarded(
        self,
        question_data: Dict[str, Any],
        validation_data: Dict[str, Any],
        reason: str
    ):
        """Save a discarded question for review."""
        record = {
            'id': str(uuid.uuid4()),
            'question_data': question_data,
            'validation_data': validation_data,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }

        # Append to discarded file and flush
        with open(self.discarded_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
            f.flush()
            os.fsync(f.fileno())

        logging.info(f"✗ Discarded question: {reason}")


class ForecastingQuestionPipeline:
    """Main pipeline orchestrating all modules."""

    def __init__(
        self,
        architect_model: str = "oss120",
        historian_model: str = "oss120",
        critic_model: str = "oss120",
        config_path: str = "models/model_servers.yaml",
        buffer_days: int = 7,
        min_resolution_buffer_days: int = 30,
        output_file: str = "generated_questions.jsonl",
        discarded_file: str = "discarded_questions.jsonl",
        checkpoint_file: str = "completed_log.txt"
    ):
        """
        Initialize the forecasting question generation pipeline.

        Args:
            architect_model: Model for taxonomy expansion
            historian_model: Model for question generation
            critic_model: Model for validation
            config_path: Path to model configuration YAML
            buffer_days: Safety buffer in days (Question_Date = Resolution_Date - buffer_days)
            min_resolution_buffer_days: Minimum days between deadline and current_date for information availability (default: 30)
            output_file: Output JSONL file for validated questions
            discarded_file: JSONL file for discarded questions
            checkpoint_file: Checkpoint file for crash recovery
        """
        self.model_config = ModelConfig(config_path)
        self.buffer_days = buffer_days
        self.min_resolution_buffer_days = min_resolution_buffer_days

        # Create LLM clients
        architect_client = self._create_llm_client(architect_model)
        historian_client = self._create_llm_client(historian_model)
        critic_client = self._create_llm_client(critic_model)

        # Initialize search engine
        self.search_engine = SearchEngine()

        # Initialize modules
        self.architect = TaxonomyExpander(architect_client)
        self.historian = QuestionHistorian(
            historian_client,
            self.search_engine,
            buffer_days,
            min_resolution_buffer_days
        )
        self.critic = QuestionCritic(critic_client, self.search_engine)

        # Initialize managers
        self.checkpoint_manager = CheckpointManager(checkpoint_file)
        self.database = QuestionDatabase(output_file, discarded_file)

        logging.info("="*80)
        logging.info("FORECASTING QUESTION GENERATION PIPELINE INITIALIZED")
        logging.info("="*80)
        logging.info(f"Architect Model: {architect_model}")
        logging.info(f"Historian Model: {historian_model}")
        logging.info(f"Critic Model: {critic_model}")
        logging.info(f"Safety Buffer: {buffer_days} days")
        logging.info(f"Min Resolution Buffer: {min_resolution_buffer_days} days (ensures information is available)")
        logging.info("="*80)

    def _create_llm_client(self, model_shortname: str) -> LLMClient:
        """Create an LLM client for a specific model."""
        server_config = self.model_config.get_server_by_shortname(model_shortname)

        if not server_config:
            raise ValueError(f"Model '{model_shortname}' not found in configuration")

        return LLMClient(
            api_base=server_config['openai_api_base'],
            api_key=server_config['openai_api_key'],
            model_name=server_config['openai_model'],
            temperature=0.7,
            max_tokens=2000
        )

    def run(
        self,
        topic: str,
        simulated_date: str,
        num_questions: int = 10,
        num_categories: int = 3,
        max_retries_per_subcategory: int = 3
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline.

        Args:
            topic: Broad topic (e.g., "Sports", "Technology")
            simulated_date: Simulated "present" date in YYYY-MM-DD format
            num_questions: Total number of questions to generate
            num_categories: Number of main categories to create
            max_retries_per_subcategory: Max attempts per subcategory before moving on

        Returns:
            Pipeline statistics
        """
        simulated_dt = datetime.strptime(simulated_date, "%Y-%m-%d")
        current_dt = datetime.now()

        logging.info("\n" + "="*80)
        logging.info("PIPELINE EXECUTION STARTED")
        logging.info("="*80)
        logging.info(f"Topic: {topic}")
        logging.info(f"Simulated Date: {simulated_date}")
        logging.info(f"Current Date: {current_dt.strftime('%Y-%m-%d')}")
        logging.info(f"Target Questions: {num_questions}")
        logging.info("="*80 + "\n")

        # Step 1: Expand taxonomy
        num_subcategories_per_category = max(2, num_questions // num_categories)
        taxonomy = self.architect.expand(
            topic,
            num_categories=num_categories,
            num_subcategories_per_category=num_subcategories_per_category
        )

        # Calculate question budget based on complexity
        total_complexity = taxonomy['total_complexity']
        category_budgets = {}

        for cat in taxonomy['categories']:
            # Allocate questions proportional to complexity
            budget = max(1, int((cat['complexity_score'] / total_complexity) * num_questions))
            category_budgets[cat['name']] = budget

        logging.info("\n" + "="*80)
        logging.info("QUESTION BUDGET ALLOCATION (by complexity)")
        logging.info("="*80)
        for cat_name, budget in category_budgets.items():
            logging.info(f"  {cat_name}: {budget} questions")
        logging.info("="*80 + "\n")

        # Step 2 & 3: Generate and validate questions
        stats = {
            'total_attempted': 0,
            'total_generated': 0,
            'total_validated': 0,
            'total_discarded': 0,
            'categories_processed': 0,
            'subcategories_processed': 0,
            'subcategories_skipped': 0
        }

        questions_generated = 0

        for category in taxonomy['categories']:
            cat_name = category['name']
            cat_complexity = category['complexity_score']
            cat_budget = category_budgets[cat_name]

            logging.info("\n" + "="*80)
            logging.info(f"[MODULE B & C: HISTORIAN + CRITIC] Category: {cat_name}")
            logging.info("="*80)
            logging.info(f"Complexity: {cat_complexity}/10 | Budget: {cat_budget} questions")
            logging.info("="*80 + "\n")

            questions_this_category = 0

            for subcategory in category['subcategories']:
                # Check if already completed
                if self.checkpoint_manager.is_completed(topic, cat_name, subcategory):
                    logging.info(f"⊘ Skipping completed subcategory: {subcategory}")
                    stats['subcategories_skipped'] += 1
                    continue

                logging.info(f"\n→ Processing subcategory: {subcategory}")

                # Try to generate questions from this subcategory
                retries = 0
                while retries < max_retries_per_subcategory and questions_this_category < cat_budget:
                    stats['total_attempted'] += 1

                    # Generate question (Historian)
                    logging.info(f"  Attempt {retries + 1}/{max_retries_per_subcategory}")
                    question_data = self.historian.generate_question(
                        topic, cat_name, subcategory, simulated_dt, current_dt
                    )

                    if question_data is None:
                        retries += 1
                        continue

                    stats['total_generated'] += 1
                    question_data['complexity_score'] = cat_complexity

                    # Validate question (Critic)
                    logging.info(f"  Validating: {question_data['question'][:60]}...")
                    is_valid, validation = self.critic.validate_question(question_data, current_dt)

                    if is_valid:
                        # Save validated question
                        self.database.save_question(question_data, validation)
                        stats['total_validated'] += 1
                        questions_this_category += 1
                        questions_generated += 1

                        if questions_generated >= num_questions:
                            logging.info("\n" + "="*80)
                            logging.info(f"✓ TARGET REACHED: {num_questions} questions generated")
                            logging.info("="*80)
                            break
                    else:
                        # Save discarded question
                        reason = f"Validation failed: {validation.get('ambiguity_reason', '')} {validation.get('leakage_reason', '')} {validation.get('timeline_reason', '')}"
                        if validation.get('answer_mismatch'):
                            reason = "Answer mismatch between Historian and Critic"

                        self.database.save_discarded(question_data, validation, reason)
                        stats['total_discarded'] += 1

                    retries += 1

                # Mark subcategory as completed
                self.checkpoint_manager.mark_completed(topic, cat_name, subcategory)
                stats['subcategories_processed'] += 1

                if questions_generated >= num_questions:
                    break

            stats['categories_processed'] += 1

            if questions_generated >= num_questions:
                break

        logging.info("\n" + "="*80)
        logging.info("PIPELINE EXECUTION COMPLETED")
        logging.info("="*80)
        logging.info(f"Questions Attempted: {stats['total_attempted']}")
        logging.info(f"Questions Generated: {stats['total_generated']}")
        logging.info(f"Questions Validated: {stats['total_validated']}")
        logging.info(f"Questions Discarded: {stats['total_discarded']}")
        logging.info(f"Success Rate: {stats['total_validated'] / max(1, stats['total_attempted']) * 100:.1f}%")
        logging.info("="*80)

        return stats
