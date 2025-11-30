# Forecasting with LLMs

A modular toolkit for LLM-powered forecasting, providing tools for knowledge gathering, analysis, and prediction.

## Available Modules

### 1. Question Generation Pipeline

Generates high-quality binary forecasting questions by "time traveling" - finding resolved past events and framing them as future uncertainties.

**Features:**
- **Three-Module Architecture**: The Architect (taxonomy expansion), The Historian (question generation), The Critic (independent validation)
- **Unbiased Question Generation**: Two-stage approach - generates questions BEFORE searching for answers to prevent hindsight bias
- **Context-Based Deduplication**: Tracks recent questions per subcategory and guides LLM to generate diverse questions
- **Genuine Uncertainty**: Prompts optimized to generate interesting questions where both Yes and No are plausible outcomes
- **Time Travel with Safety**: Backdates questions with 7-day buffer and enforces 30-day resolution buffer for information availability
- **Crash-Proof**: Checkpoint-based recovery allows 24+ hour runs to resume from interruptions
- **Free Search**: Uses DuckDuckGo (no API keys required) for answer verification

### 2. News Retrieval Pipeline

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

### Question Generation Pipeline

#### Basic Example

```python
from forecast_kag.question_generation_pipeline import ForecastingQuestionPipeline

# Initialize pipeline with different models for each module
pipeline = ForecastingQuestionPipeline(
    architect_model="alcf_oss20b",      # Model for taxonomy expansion
    historian_model="alcf_oss20b",      # Model for question generation
    critic_model="alcf_oss20b",         # Model for validation
    config_path="models/model_servers.yaml",
    buffer_days=7,                      # Safety buffer between question and resolution
    min_resolution_buffer_days=30,      # Minimum days since deadline for info availability
    output_file="data/generated_questions.jsonl",
    discarded_file="data/discarded_questions.jsonl",
    checkpoint_file="logs/completed_log.txt"
)

# Run pipeline
stats = pipeline.run(
    topic="Technology",              # Broad topic to explore
    simulated_date="2024-07-01",     # The "present" we're simulating
    num_questions=50,                # How many questions to generate
    num_categories=5                 # Number of main categories
)
```

#### Pipeline Parameters

**Model Selection:**
- `architect_model` (str): Model for taxonomy expansion (default: "oss120")
- `historian_model` (str): Model for question generation (default: "oss120")
- `critic_model` (str): Model for independent validation (default: "oss120")

**Core Parameters:**
- `config_path` (str): Path to model config YAML (default: "models/model_servers.yaml")
- `buffer_days` (int): Safety buffer in days between question date and resolution date (default: 7)
- `min_resolution_buffer_days` (int): Minimum days between deadline and current date for information availability (default: 30)
- `output_file` (str): Output JSONL file for validated questions (default: "generated_questions.jsonl")
- `discarded_file` (str): JSONL file for questions that failed validation (default: "discarded_questions.jsonl")
- `checkpoint_file` (str): Checkpoint file for crash recovery (default: "completed_log.txt")

#### Run Method

```python
stats = pipeline.run(
    topic: str,           # Broad topic (e.g., "Sports", "Politics", "Technology")
    simulated_date: str,  # "YYYY-MM-DD" - The simulated "present" for question generation
    num_questions: int,   # Target number of validated questions to generate
    num_categories: int   # Number of main categories to expand from the topic
)
```

#### Statistics Dictionary

```python
{
    'total_attempted': int,          # Total question generation attempts
    'total_generated': int,          # Questions successfully generated
    'total_validated': int,          # Questions passing validation
    'total_discarded': int,          # Questions rejected by Critic
    'categories_processed': int,     # Number of categories processed
    'subcategories_processed': int,  # Number of subcategories processed
    'subcategories_skipped': int     # Subcategories skipped (checkpoint recovery)
}
```

#### Output Format (JSONL)

Each validated question is saved as a JSON line in the output file:

```json
{
  "id": "uuid_v4_string",
  "topic": "Technology",
  "category": "Consumer Electronics",
  "subcategory": "iPhone Releases",
  "question": "Will Apple release iPhone 16 before September 30, 2024?",
  "t_ask": "2024-03-15",
  "t_resolve": "2024-03-22",
  "answer": "Yes",
  "meta": {
    "complexity_score": 8,
    "validation_confidence": "High",
    "event_description": "Apple released iPhone 16 on September 20, 2024"
  }
}
```

**Key Properties:**
- `t_ask`: Question date (Resolution date - buffer_days)
- `t_resolve`: Actual event resolution date
- `answer`: Ground truth ("Yes" or "No")
- No news context saved (prevents model bias in downstream training)

#### Context System

The pipeline automatically tracks recently generated questions within each subcategory to prevent repetition:

- Stores last 10 questions per subcategory (configurable via `max_context_size`)
- Injects previous questions into prompt with instructions to generate different questions
- Encourages diversity in entities, timeframes, and actions

**Example Context Injection:**
```
PREVIOUSLY GENERATED QUESTIONS FOR THIS SUBCATEGORY:
DO NOT generate questions similar to these. Choose DIFFERENT entities, events, and timeframes:
1. Will Gavin Newsom win the California gubernatorial election before November 5, 2024?
2. Will Ron DeSantis win the Florida gubernatorial election before November 8, 2024?

Generate a UNIQUE question that is DIFFERENT from all of the above.
```

### News Retrieval Pipeline

#### Basic Example

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

