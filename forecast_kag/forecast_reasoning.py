"""
Reasoning + Forecast Aggregation Pipeline for Forecasting
Uses LLMs to apply scratchpad reasoning to news summaries and generate predictions.
"""

import os
import yaml
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from openai import OpenAI


class ModelConfig:
    def __init__(self, config_path: str = "models/model_servers.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.servers = cfg.get("servers", [])

    def get_server(self, shortname: str) -> Optional[Dict[str, str]]:
        for s in self.servers:
            if s["shortname"] == shortname:
                return s
        return None


class LLMClient:
    def __init__(self, api_base: str, api_key: str, model_name: str,
                 temperature: float = 0.7, max_tokens: int = 1500):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        temp = temperature if temperature is not None else self.temperature
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content


class ReasoningAgent:
    """
    Applies scratchpad reasoning to:
      - question
      - background
      - resolution criteria
      - news summaries
    and produces a probability forecast.
    """

    def __init__(self,
                 llm_client: LLMClient,
                 scratchpad_prompt_path: str = "forecast_kag/prompts/forecast_reasoning_scratchpad.txt"):
        self.llm_client = llm_client

        with open(scratchpad_prompt_path, "r") as f:
            self.scratchpad_template = f.read()

    def build_prompt(self,
                     question: str,
                     background: str,
                     resolution_criteria: str,
                     date_begin: str,
                     date_end: str,
                     retrieved_info: str) -> str:

        return self.scratchpad_template.format(
            question=question,
            background=background,
            resolution_criteria=resolution_criteria,
            date_begin=date_begin,
            date_end=date_end,
            retrieved_info=retrieved_info
        )

    def run_reasoning(self,
                      question: str,
                      background: str,
                      resolution_criteria: str,
                      date_begin: str,
                      date_end: str,
                      retrieved_info: str) -> Dict[str, Any]:

        prompt = self.build_prompt(
            question=question,
            background=background,
            resolution_criteria=resolution_criteria,
            date_begin=date_begin,
            date_end=date_end,
            retrieved_info=retrieved_info
        )

        raw_output = self.llm_client.generate(prompt)

        prob = self._extract_probability(raw_output)

        return {
            "raw_reasoning": raw_output,
            "forecast": prob
        }

    def _extract_probability(self, text: str) -> float:
        """
        Extract the final "*0.xxx*" style value.
        Default fallback = 0.5
        """
        import re

        match = re.search(r"\*(0?\.\d+)\*", text)
        if match:
            try:
                return float(match.group(1))
            except:
                return 0.5

        nums = re.findall(r"0?\.\d+", text)
        if nums:
            try:
                return float(nums[-1])
            except:
                pass

        return 0.5


class AggregationAgent:
    """Simple trimmed mean aggregator."""

    def __init__(self, trim_fraction: float = 0.15):
        self.trim = trim_fraction

    def aggregate(self, predictions: List[float]) -> float:
        if not predictions:
            return 0.5

        preds = sorted(predictions)
        n = len(preds)
        k = int(self.trim * n)

        trimmed = preds[k:n-k] if n > 2 * k else preds
        return sum(trimmed) / len(trimmed)


class ForecastReasoningPipeline:
    """
    Combines:
      - news retrieval output
      - scratchpad reasoning
      - ensemble aggregation
    """

    def __init__(self,
                 model_shortname: str,
                 config_path: str = "models/model_servers.yaml",
                 scratchpad_prompt_path: str = "forecast_kag/prompts/forecast_reasoning_scratchpad.txt"):

        mc = ModelConfig(config_path)
        model_cfg = mc.get_server(model_shortname)

        if not model_cfg:
            raise ValueError(f"Model shortname '{model_shortname}' not found.")

        self.client = LLMClient(
            api_base=model_cfg["openai_api_base"],
            api_key=model_cfg["openai_api_key"],
            model_name=model_cfg["openai_model"],
            temperature=0.7
        )

        self.reasoner = ReasoningAgent(
            llm_client=self.client,
            scratchpad_prompt_path=scratchpad_prompt_path
        )
        self.aggregator = AggregationAgent(trim_fraction=0.15)

    def run(self,
            question: str,
            background: str,
            resolution_criteria: str,
            date_begin: str,
            date_end: str,
            summaries: List[str]) -> Dict[str, Any]:

        retrieved_info = "\n\n".join(f"- {s}" for s in summaries)

        outputs = []
        predictions = []

        for i in range(3):
            result = self.reasoner.run_reasoning(
                question=question,
                background=background,
                resolution_criteria=resolution_criteria,
                date_begin=date_begin,
                date_end=date_end,
                retrieved_info=retrieved_info
            )
            outputs.append(result)
            predictions.append(result["forecast"])

        final_prediction = self.aggregator.aggregate(predictions)

        return {
            "individual_outputs": outputs,
            "ensemble_prediction": final_prediction
        }