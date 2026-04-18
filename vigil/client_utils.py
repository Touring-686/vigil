import logging
import os

import openai

logger = logging.getLogger(__name__)


def create_openai_client_for_model(model_name: str) -> openai.OpenAI:
    if model_name.lower().startswith("gemini-") or model_name.upper().startswith("GEMINI_"):
        # 使用GOOGLE API配置
        google_api_key = os.getenv("GOOGLE_API_KEY")
        google_base_url = os.getenv("GOOGLE_BASE_URL")

        if not google_api_key:
            logger.warning(
                f"[VIGIL] GOOGLE_API_KEY not set for GEMINI model {model_name}, "
                f"falling back to default OpenAI client"
            )
            return openai.OpenAI()

        if google_base_url:
            logger.info(f"[VIGIL] Using GEMINI model via OpenAI-compatible interface at {google_base_url}")
            return openai.OpenAI(
                api_key=google_api_key,
                base_url=google_base_url,
            )
        else:
            logger.warning(
                f"[VIGIL] GOOGLE_BASE_URL not set for GEMINI model {model_name}, "
                f"using default base_url"
            )
            return openai.OpenAI(api_key=google_api_key)
    else:
        return openai.OpenAI()
