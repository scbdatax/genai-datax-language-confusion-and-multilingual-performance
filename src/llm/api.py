import asyncio
from typing import List
from openai import OpenAI
from together import Together
from together import AsyncTogether
from src.constants import TYPHOON_API_KEY
from src.constants import TYPHOON_ENDPOINT
from src.constants import TOGETHER_API_KEY
from src.constants import OPENAI_API_KEY
from src.constants import OPENAI_ORGANIZATION_ID


# TODO: apply `asyncio` from https://stackoverflow.com/questions/69161694/asyncio-can-i-wrap-a-sync-rest-call-in-async
class LLM:
    def __init__(
        self, 
        client_name: str, 
        model_name: str,
    ) -> None:
        self.client_name = client_name
        self.model_name = model_name
        self._init_client()

    def enjoy(
        self,
        article: str, 
        prompt_lang: str,
        summary_lang: str,
        max_tokens: int = 1000,
        temperature: float = 0.01,
        top_p: float = 1,
        stream: bool = False,
    ) -> str:
        content = self.prepare_content(article, prompt_lang, summary_lang)
        response = self.get_response(
            content,
            max_tokens,
            temperature,
            top_p,
            stream,
        )
        return response

    def _init_client(self) -> None:
        if self.client_name == "typhoon":
            self.client = OpenAI(
                api_key=TYPHOON_API_KEY,
                base_url=TYPHOON_ENDPOINT,
            )
        elif self.client_name == "together":
            self.client = Together(api_key=TOGETHER_API_KEY)

        elif self.client_name == "together-async":
            self.client = AsyncTogether(api_key=TOGETHER_API_KEY)

        elif self.client_name == "openai":
            self.client = OpenAI(
                api_key=OPENAI_API_KEY,
                organization=OPENAI_ORGANIZATION_ID,
            )

    @staticmethod
    def prepare_content(
        article: str, 
        prompt_lang: str,
        summary_lang: str,
    ) -> str:

        prompt_template = {
            "en": "Article: {article}\n\nSummarize the article above in {lang} paragraph with at least 100 words.\n\nSummary: ",
            "th": "บทความ: {article}\n\nสรุปบทความข้างต้นเป็นภาษา{lang}ใน 1 ย่อหน้าโดยใช้จำนวนคำอย่างต่ำ 100 คำ\n\nสรุป: ",
        }

        lang_mapping = {
            "en": {
                "en": "English",
                "th": "Thai",
            },
            "th": {
                "en": "อังกฤษ",
                "th": "ไทย",
            },
        }

        return prompt_template[prompt_lang].format(
            article=article,
            lang=lang_mapping[prompt_lang][summary_lang],
        )

    def get_response(
        self,
        content: str,
        max_tokens: int = 1000,
        temperature: float = 0.01,
        top_p: float = 1,
        stream: bool = False,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": content,
            }],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
        )
        return response.choices[0].message.content

    async def get_response_async(
        self,
        contents: List[str],
        max_tokens: int = 1000,
        temperature: float = 0.01,
        top_p: float = 1,
        stream: bool = False,
    ) -> List[str]:
        tasks = [
            self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": content,
                }],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=stream,
            )
            for content in contents
        ]
        responses = await asyncio.gather(*tasks)
        responses = [
            response.choices[0].message.content
            for response in responses
        ]
        return responses
