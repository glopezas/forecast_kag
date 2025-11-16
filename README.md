# Forecasting with LLMs

A modular toolkit for LLM-powered forecasting, providing tools for knowledge gathering, analysis, and prediction.

## Available Modules

### News Retrieval Pipeline (Implemented)

Retrieves and synthesizes relevant news articles for forecasting questions.

**Features:**
- **Keyword Generation**: Automatically generates search keywords optimized for Google News
- **News Retrieval**: Fetches full article content from Google News using improved redirect resolution
- **Relevance Rating**: Rates articles on a 1-5 scale based on relevance to the forecasting question
- **News Synthesis**: Combines all relevant articles into a single cohesive summary of key factors

### Coming Soon

Additional forecasting modules are planned for future releases.

## Installation

### 1. Create Conda Environment

```bash
conda create -n forecast_kag python=3.11
conda activate forecast_kag
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

## Configuration

### Model Configuration File

Create a YAML file at `models/model_servers.yaml` with your LLM server configurations:

```yaml
servers:
  - shortname: llama70
    openai_api_base: http://localhost:8000/v1
    openai_api_key: dummy-key
    openai_model: meta-llama/Llama-3.1-70B-Instruct

  - shortname: gpt4
    openai_api_base: https://api.openai.com/v1
    openai_api_key: sk-your-actual-key-here
    openai_model: gpt-4-turbo-preview
```

**Required fields for each server:**
- `shortname`: Identifier you'll use to reference this model
- `openai_api_base`: Base URL for the OpenAI-compatible API endpoint
- `openai_api_key`: API key for authentication
- `openai_model`: Model name/identifier

## Usage

### Basic Example

```python
from forecast_kag.news_retrieval import NewsRetrievalPipeline

# Initialize pipeline
pipeline = NewsRetrievalPipeline(
    model_shortname="llama70",       # From your models/model_servers.yaml
    config_path="models/model_servers.yaml",  # Path to config file
    num_questions=5,                 # Number of search keywords to generate
    news_per_keyword=6,              # Articles to retrieve per keyword
    min_news_rating=3,               # Minimum rating threshold (1-5)
    news_period_days=90,             # How far back to search for news
    question_gen_temp=0.7,           # Temperature for keyword generation
    news_rating_temp=0.3,            # Temperature for news rating
    summarization_temp=0.5,          # Temperature for summarization
    max_tokens=1000                  # Max tokens for LLM responses
)

# Run pipeline
results = pipeline.run(
    question="Will Apple release a new iPhone model before June 2024?",
    background="Apple typically releases new iPhone models in September each year.",
    question_date="2024-03-01"  # Cutoff date for news retrieval
)

# Access results
print(results['summary'])                  # Synthesized summary
print(results['search_keywords'])          # Generated keywords
print(results['relevant_news'])            # Articles that passed threshold
print(results['stats'])                    # Pipeline statistics
```

### Pipeline Parameters

**Core Parameters:**
- `model_shortname` (str): Model identifier from your config file
- `config_path` (str): Path to model config YAML (default: "models/model_servers.yaml")
- `num_questions` (int): Number of search keywords to generate (default: 5)
- `news_per_keyword` (int): Articles to retrieve per keyword (default: 6)
- `min_news_rating` (int): Minimum relevance rating 1-5 (default: 3)
- `news_period_days` (int): Days to search back for news (default: 90)

**LLM Parameters:**
- `question_gen_temp` (float): Temperature for keyword generation (default: 0.7)
- `news_rating_temp` (float): Temperature for rating (default: 0.3)
- `summarization_temp` (float): Temperature for synthesis (default: 0.5)
- `max_tokens` (int): Max tokens for LLM responses (default: 1000)

### Run Method

```python
results = pipeline.run(
    question: str,        # Main forecasting question
    background: str = "", # Optional background context
    question_date: str = None  # "YYYY-MM-DD" format for news cutoff
)
```

### Results Dictionary

```python
{
    'question': str,              # Original question
    'background': str,            # Background information
    'question_date': str,         # Question date
    'search_keywords': List[str], # Generated search keywords
    'all_rated_news': List[Dict], # All articles with ratings
    'relevant_news': List[Dict],  # Articles >= min_rating threshold
    'summary': str,               # Synthesized summary (2-4 paragraphs)
    'stats': {
        'num_search_keywords': int,
        'total_articles_retrieved': int,
        'total_articles_rated': int,
        'relevant_articles': int,
        'min_rating_threshold': int
    }
}
```

## Pipeline Architecture

### 1. Keyword Generation Agent
- **Input**: Forecasting question + background
- **Process**: Uses LLM to generate targeted search keywords (3-7 words each)
- **Output**: List of optimized search phrases

### 2. News Retrieval Agent
- **Input**: Search keywords + date range
- **Process**:
  - Searches Google News for each keyword
  - Resolves Google News redirects using internal API
  - Fetches full article content with newspaper3k
  - Validates content length (min 100 chars)
- **Output**: Articles with full content
- **Note**: Discards articles where fetching fails

### 3. News Rating Agent
- **Input**: Articles + forecasting question
- **Process**: Rates each article's relevance (1-5 scale)
- **Output**: Rated articles sorted by relevance
- **Prompt**: `forecast_kag/prompts/news_rating.txt`

### 4. Summarization Agent
- **Input**: Relevant articles (rating >= threshold)
- **Process**: Synthesizes all articles into single cohesive summary
- **Output**: 2-4 paragraph factual analysis
- **Prompt**: `forecast_kag/prompts/news_synthesis.txt`



## Examples

See `news_retrieval_demo.ipynb` for a complete Jupyter notebook demonstration.

