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

# Use different models for different agents
pipeline = NewsRetrievalPipeline(
    keyword_model="llama70",         # Model for keyword generation
    rating_model="qwen80",           # Model for rating articles
    summarization_model="oss120",    # Model for summarization
    config_path="models/model_servers.yaml",
    num_keywords=5,
    news_per_keyword=6,
    min_news_rating=3,
    news_period_days=90,
    question_gen_temp=0.7,
    news_rating_temp=0.3,
    summarization_temp=0.5,
    max_tokens=1000
)

# Run pipeline
results = pipeline.run(
    question="Will Apple release a new iPhone model before June 2024?",
    background="Apple typically releases new iPhone models in September each year.",
    question_date="2024-03-01"  # Cutoff date for news retrieval
)
```

### Pipeline Parameters

**Model Selection:**
- `keyword_model` (str): Model for keyword generation
- `rating_model` (str): Model for article rating
- `summarization_model` (str): Model for synthesis

**Core Parameters:**
- `config_path` (str): Path to model config YAML (default: "models/model_servers.yaml")
- `num_keywords` (int): Number of search keywords to generate (default: 5)
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

## Examples

See `news_retrieval_demo.ipynb` for a complete Jupyter notebook demonstration.

