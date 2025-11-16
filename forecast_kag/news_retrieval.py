"""
News Retrieval Pipeline for Forecasting
Uses LLMs to generate search keywords, retrieve news, rate relevance, and synthesize findings.
"""

import yaml
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import re

from openai import OpenAI
from gnews import GNews
from newspaper import Article
import requests
import json
from bs4 import BeautifulSoup

# Configure logging - suppress HTTP request logs but show pipeline progress
logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class ModelConfig:
    """Load and manage model configurations from YAML file."""

    def __init__(self, config_path: str = "models/model_servers.yaml"):
        """
        Initialize model configuration.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.servers = self._load_config()

    def _load_config(self) -> List[Dict[str, str]]:
        """Load server configurations from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('servers', [])

    def get_server_by_shortname(self, shortname: str) -> Optional[Dict[str, str]]:
        """
        Get server configuration by shortname.

        Args:
            shortname: Short identifier for the server (e.g., 'llama70')

        Returns:
            Server configuration dictionary or None if not found
        """
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
        max_tokens: int = 1000
    ):
        """
        Initialize LLM client.

        Args:
            api_base: Base URL for the API endpoint
            api_key: API key for authentication
            model_name: Name of the model to use
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, temperature: Optional[float] = None, log_context: str = None) -> str:
        """
        Generate text using the LLM.

        Args:
            prompt: Input prompt
            temperature: Override default temperature if provided
            log_context: Context for logging (e.g., "Question Generation")

        Returns:
            Generated text
        """
        temp = temperature if temperature is not None else self.temperature

        if log_context:
            logging.info(f"Calling LLM for {log_context}...")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                max_tokens=self.max_tokens
            )

            if log_context:
                logging.info(f"{log_context} completed")

            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)
            logging.error(f"LLM API Error: {error_msg[:200]}")
            raise RuntimeError(f"Failed to get response from LLM server at {self.api_base}. "
                             f"Error: {error_msg[:200]}. "
                             f"Please check that your server is running and accessible.") from e


class KeywordGenerationAgent:
    """Agent for generating search keywords from a main forecasting question."""

    def __init__(self, llm_client: LLMClient, prompt_path: str = "forecast_kag/prompts/question_generation.txt"):
        self.llm_client = llm_client
        with open(prompt_path, 'r') as f:
            self.prompt_template = f.read()

    def generate_keywords(self, question: str, background: str = "", num_keywords: int = 5) -> List[str]:
        logging.info(f"[Agent 1: Keyword Generation] Generating {num_keywords} search keywords...")

        prompt = self.prompt_template.format(
            question=question,
            background=background if background else "No additional background provided.",
            num_questions=num_keywords
        )

        response = self.llm_client.generate(prompt, temperature=0.7, log_context="Keyword Generation")

        keywords = [re.match(r'^\d+[\.\)]\s*(.+)$', line.strip()).group(1)
                   for line in response.strip().split('\n')
                   if re.match(r'^\d+[\.\)]\s*(.+)$', line.strip())]

        logging.info(f"Generated {len(keywords)} search keywords")
        return keywords[:num_keywords]


class NewsRetrievalAgent:
    """Agent for retrieving news articles using GNews with date filtering support."""

    def __init__(
        self,
        max_results: int = 6,
        language: str = "en"
    ):
        """
        Initialize news retrieval agent.

        Args:
            max_results: Maximum number of articles to retrieve per query
            language: Language code for news articles (e.g., 'en')
        """
        self.max_results = max_results
        self.language = language

        # Initialize GNews client (no country filter for global news)
        self.gnews = GNews(
            language=language,
            max_results=max_results
        )

    def search_news(
        self,
        query: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for news articles using GNews with date filtering.

        Args:
            query: Search query
            start_date: Get articles published after this date (inclusive)
            end_date: Get articles published before this date (inclusive)

        Returns:
            List of news article dictionaries
        """
        try:
            # According to documentation, dates can be set as properties on the instance
            # Create a new GNews instance and set dates as properties
            if start_date or end_date:
                start_tuple = (start_date.year, start_date.month, start_date.day) if start_date else None
                end_tuple = (end_date.year, end_date.month, end_date.day) if end_date else None

                logging.debug(f"  GNews search: query='{query[:50]}...', start={start_tuple}, end={end_tuple}")

                # Create GNews instance
                gnews_filtered = GNews(
                    language=self.language,
                    max_results=self.max_results
                )

                # Set dates as properties (as shown in documentation)
                gnews_filtered.start_date = start_tuple
                gnews_filtered.end_date = end_tuple

                articles = gnews_filtered.get_news(query)
                gnews_instance = gnews_filtered
            else:
                # Use default instance without date filters
                articles = self.gnews.get_news(query)
                gnews_instance = self.gnews

            # Fetch full article content for each article
            articles = self._fetch_full_articles(articles, gnews_instance)

            # Additional safety filter to ensure no future articles
            if start_date or end_date:
                articles = self._filter_by_date(articles, start_date, end_date)

            return articles

        except Exception as e:
            logging.error(f"Error retrieving news for query '{query}': {e}")
            return []

    def _get_redirect_url(self, google_news_url: str) -> Optional[str]:
        try:
            resp = requests.get(google_news_url, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            c_wiz = soup.select_one('c-wiz[data-p]')

            if not c_wiz:
                return None

            data = c_wiz.get('data-p')
            obj = json.loads(data.replace('%.@.', '["garturlreq",'))

            payload = {'f.req': json.dumps([[['Fbv4je', json.dumps(obj[:-6] + obj[-2:]), 'null', 'generic']]])}
            headers = {
                'content-type': 'application/x-www-form-urlencoded;charset=UTF-8',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            }

            response = requests.post("https://news.google.com/_/DotsSplashUi/data/batchexecute",
                                   headers=headers, data=payload, timeout=10)
            array_string = json.loads(response.text.replace(")]}'", ""))[0][2]
            return json.loads(array_string)[1]

        except Exception:
            return None

    def _fetch_full_articles(self, articles: List[Dict], gnews_instance: GNews) -> List[Dict]:
        """Fetch full article content. Discards articles where fetching fails."""
        enriched_articles = []
        discarded_count = 0

        for article in articles:
            try:
                url = article.get('url')
                if not url:
                    discarded_count += 1
                    continue

                actual_url = self._get_redirect_url(url) if 'news.google.com' in url else url
                if not actual_url:
                    discarded_count += 1
                    continue

                full_article = Article(actual_url, language=self.language)
                full_article.download()
                full_article.parse()

                article_text = getattr(full_article, 'text_cleaned', None) or getattr(full_article, 'text', None)

                if article_text and len(article_text) >= 100:
                    article['full_content'] = article_text
                    article['authors'] = getattr(full_article, 'authors', [])
                    enriched_articles.append(article)
                else:
                    discarded_count += 1

            except Exception:
                discarded_count += 1

        if discarded_count > 0:
            logging.info(f"Discarded {discarded_count}/{len(articles)} articles")

        return enriched_articles

    def _filter_by_date(self, articles: List[Dict], start_date: Optional[datetime], end_date: Optional[datetime]) -> List[Dict]:
        filtered = []
        for article in articles:
            try:
                pub_date = article.get('published date')
                if not pub_date:
                    continue

                article_date = self._parse_date(pub_date)

                if start_date and article_date < start_date:
                    continue
                if end_date and article_date > end_date:
                    logging.warning(f"Excluded FUTURE article: '{article.get('title', 'N/A')[:50]}'")
                    continue

                filtered.append(article)
            except Exception:
                continue

        return filtered

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object."""
        from dateutil import parser
        parsed = parser.parse(date_str)
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed

    def retrieve_for_questions(
        self,
        questions: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve news for multiple questions.

        Args:
            questions: List of search queries
            start_date: Filter articles after this date
            end_date: Filter articles before this date

        Returns:
            Dictionary mapping questions to their retrieved articles
        """
        logging.info(f"Executing {len(questions)} news searches...")
        results = {}

        for i, question in enumerate(questions, 1):
            query_type = "[ORIGINAL]" if i == 1 else f"[KEYWORD {i-1}]"
            logging.info(f"Query {i}/{len(questions)} {query_type}: '{question[:60]}...'")
            articles = self.search_news(question, start_date, end_date)
            results[question] = articles
            logging.info(f"Found {len(articles)} articles")

        total_articles = sum(len(articles) for articles in results.values())
        logging.info(f"Retrieved {total_articles} total articles from {len(questions)} queries")
        return results


class NewsRatingAgent:
    """Agent for rating news article relevance."""

    def __init__(self, llm_client: LLMClient, prompt_path: str = "forecast_kag/prompts/news_rating.txt"):
        self.llm_client = llm_client
        with open(prompt_path, 'r') as f:
            self.prompt_template = f.read()

    def rate_article(self, question: str, article: Dict[str, Any]) -> int:
        content = article.get('full_content', '') or article.get('description', 'No description')
        if len(content) > 3000:
            content = content[:3000] + "... [truncated]"

        prompt = self.prompt_template.format(
            question=question,
            news_title=article.get('title', 'No title'),
            news_description=content,
            news_date=article.get('published date', 'Unknown date')
        )

        response = self.llm_client.generate(prompt, temperature=0.3, log_context=None)

        match = re.search(r'([1-5])', response)
        return int(match.group(1)) if match else 3



class NewsSummarizationAgent:
    """Agent for synthesizing relevant news articles."""

    def __init__(self, llm_client: LLMClient, synthesis_prompt_path: str = "forecast_kag/prompts/news_synthesis.txt"):
        self.llm_client = llm_client
        with open(synthesis_prompt_path, 'r') as f:
            self.synthesis_template = f.read()

    def _format_articles_for_synthesis(self, articles: List[Dict[str, Any]]) -> str:
        formatted = []
        for i, article in enumerate(articles, 1):
            content = article.get('full_content', '') or article.get('description', 'No content available')
            if len(content) > 2000:
                content = content[:2000] + "... [truncated]"

            formatted.append(f"Article {i}:\nTitle: {article.get('title', 'No title')}\n"
                           f"Date: {article.get('published date', 'Unknown date')}\nContent: {content}\n")

        return '\n---\n'.join(formatted)

    def summarize_news(self, question: str, background: str, articles: List[Dict[str, Any]]) -> str:
        logging.info(f"\n[Agent 4: Summarization] Synthesizing {len(articles)} articles into cohesive summary...")

        prompt = self.synthesis_template.format(
            question=question,
            background=background if background else "No additional background.",
            num_articles=len(articles),
            articles=self._format_articles_for_synthesis(articles)
        )

        response = self.llm_client.generate(prompt, temperature=0.5, log_context="Article Synthesis")
        logging.info(f"Synthesis completed ({len(response)} characters)")
        return response.strip()


class NewsRetrievalPipeline:
    """Main pipeline orchestrating all agents for news retrieval and analysis."""

    def __init__(
        self,
        model_shortname: str,
        config_path: str = "models/model_servers.yaml",
        num_keywords: int = 5,
        news_per_keyword: int = 6,
        min_news_rating: int = 3,
        news_period_days: int = 90,
        question_gen_temp: float = 0.7,
        news_rating_temp: float = 0.3,
        summarization_temp: float = 0.5,
        max_tokens: int = 1000
    ):
        """
        Initialize the news retrieval pipeline.

        Args:
            model_shortname: Shortname of the model to use (from config)
            config_path: Path to model configuration YAML
            num_keywords: Number of search keywords to generate
            news_per_keyword: Number of news articles to retrieve per keyword
            min_news_rating: Minimum rating to keep articles (1-5)
            news_period_days: Number of days to search back for news
            question_gen_temp: Temperature for keyword generation
            news_rating_temp: Temperature for news rating
            summarization_temp: Temperature for summarization
            max_tokens: Maximum tokens for LLM generation
        """
        # Load model configuration
        self.model_config = ModelConfig(config_path)
        server_config = self.model_config.get_server_by_shortname(model_shortname)

        if not server_config:
            raise ValueError(f"Model '{model_shortname}' not found in configuration")

        # Create LLM client
        self.llm_client = LLMClient(
            api_base=server_config['openai_api_base'],
            api_key=server_config['openai_api_key'],
            model_name=server_config['openai_model'],
            temperature=0.7,
            max_tokens=max_tokens
        )

        # Initialize agents
        self.keyword_agent = KeywordGenerationAgent(self.llm_client)

        # Use GNews for news retrieval
        self.news_agent = NewsRetrievalAgent(
            max_results=news_per_keyword,
            language="en"
        )

        self.rating_agent = NewsRatingAgent(self.llm_client)
        self.summary_agent = NewsSummarizationAgent(self.llm_client)

        # Store parameters
        self.num_keywords = num_keywords
        self.news_per_keyword = news_per_keyword
        self.min_news_rating = min_news_rating
        self.news_period_days = news_period_days

    def run(
        self,
        question: str,
        background: str = "",
        question_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete news retrieval pipeline.

        Args:
            question: Main forecasting question
            background: Background information about the question
            question_date: Date of the question (YYYY-MM-DD format)
                          News will be retrieved before this date

        Returns:
            Dictionary containing:
                - question: Original question
                - background: Background information
                - question_date: Question date
                - search_keywords: Generated search keywords used for news retrieval
                - all_rated_news: ALL retrieved articles with their ratings (sorted by rating)
                - relevant_news: Only articles with rating >= min_threshold (sorted by rating)
                - summary: Synthesized summary of all relevant news (single cohesive text)
                - stats: Pipeline statistics
        """
        logging.info("="*80)
        logging.info("NEWS RETRIEVAL PIPELINE")
        logging.info("="*80)
        logging.info(f"Question: {question}")
        logging.info(f"Date Range: {self.news_period_days} days")
        logging.info("="*80)

        # Parse question date
        end_date = None
        if question_date:
            try:
                end_date = datetime.strptime(question_date, "%Y-%m-%d")
                logging.info(f"News search cutoff date: {question_date}")
            except:
                logging.warning(f"Could not parse date '{question_date}', using current date")
                end_date = datetime.now()
        else:
            end_date = datetime.now()

        start_date = end_date - timedelta(days=self.news_period_days)
        logging.info(f"Searching news from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        # Step 1: Generate search keywords
        search_keywords = self.keyword_agent.generate_keywords(
            question=question,
            background=background,
            num_keywords=self.num_keywords
        )
        for i, kw in enumerate(search_keywords, 1):
            logging.info(f"  {i}. {kw}")

        all_search_queries = [question] + search_keywords
        logging.info(f"\n[Agent 2: News Retrieval] Preparing search queries...")
        logging.info(f"Original question: '{question[:80]}...'")
        logging.info(f"Total search queries: {len(all_search_queries)} (1 original + {len(search_keywords)} keywords)")

        news_by_question = self.news_agent.retrieve_for_questions(
            questions=all_search_queries,
            start_date=start_date,
            end_date=end_date
        )

        all_articles = []
        future_articles_caught = 0

        for q, articles in news_by_question.items():
            for article in articles:
                article['search_query'] = q

                pub_date = article.get('published date')
                if pub_date:
                    try:
                        article_date = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z") if isinstance(pub_date, str) else pub_date
                        if hasattr(article_date, 'strftime'):
                            if article_date > end_date:
                                future_articles_caught += 1
                                logging.error(f"CRITICAL: Caught future article in final check: '{article.get('title', 'N/A')[:50]}' (published {pub_date}, cutoff {end_date.strftime('%Y-%m-%d')})")
                                continue
                    except:
                        pass

                all_articles.append(article)

        if future_articles_caught > 0:
            logging.error(f"CRITICAL: Blocked {future_articles_caught} future articles in final validation")

        logging.info(f"\n[Agent 3: Rating] Rating {len(all_articles)} articles...")

        all_rated_articles = []

        for i, article in enumerate(all_articles, 1):
            logging.info(f"\nArticle {i}/{len(all_articles)}: '{article.get('title', 'No title')[:60]}...'")
            logging.info(f"Rating...")

            rating = self.rating_agent.rate_article(question, article)
            article['relevance_rating'] = rating
            all_rated_articles.append(article)

            logging.info(f"Rating: {rating}/5")

            if rating < self.min_news_rating:
                logging.info(f"Filtered out (rating < {self.min_news_rating})")

        all_rated_articles.sort(key=lambda x: x['relevance_rating'], reverse=True)

        filtered_articles = [a for a in all_rated_articles if a['relevance_rating'] >= self.min_news_rating]

        if filtered_articles:
            logging.info(f"\n{len(filtered_articles)} articles meet threshold (rating >= {self.min_news_rating})")
            summary = self.summary_agent.summarize_news(
                question=question,
                background=background,
                articles=filtered_articles
            )
        else:
            logging.info(f"\nNo articles met the minimum rating threshold of {self.min_news_rating}")
            summary = "No relevant news articles found for this forecasting question."

        logging.info("\n" + "="*80)
        logging.info("PIPELINE COMPLETED SUCCESSFULLY")
        logging.info("="*80)

        return {
            'question': question,
            'background': background,
            'question_date': question_date,
            'search_keywords': search_keywords,
            'all_rated_news': all_rated_articles,  # ALL articles with their ratings
            'relevant_news': filtered_articles,     # Only articles with rating >= threshold
            'summary': summary,                     # Synthesized summary of relevant news
            'stats': {
                'num_search_keywords': len(search_keywords),
                'total_articles_retrieved': len(all_articles),
                'total_articles_rated': len(all_rated_articles),
                'relevant_articles': len(filtered_articles),
                'min_rating_threshold': self.min_news_rating
            }
        }


def load_available_models(config_path: str = "models/model_servers.yaml") -> List[str]:
    """
    Load list of available model shortnames from configuration.

    Args:
        config_path: Path to model configuration YAML

    Returns:
        List of available model shortnames
    """
    config = ModelConfig(config_path)
    return config.get_all_shortnames()
