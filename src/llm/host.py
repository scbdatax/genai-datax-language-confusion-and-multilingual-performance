import torch
from vllm import LLM, SamplingParams

class vLLM:

    model_path_dict = {
        "sailor": "sail/Sailor-7B-Chat",
        "qwen15": "Qwen/Qwen1.5-7B",
        "qwen25": "Qwen/Qwen2.5-7B",
        "otg": "openthaigpt/openthaigpt1.5-7b-instruct",
    }

    def __init__(
        self, 
        model_name: str,
    ) -> None:
        self.model_name = model_name
        model_path = self.model_path_dict[model_name]
        self.llm = self._load_model(model_path)

    @staticmethod
    def _load_model(model_path: str) -> LLM:
        llm = LLM(
            model_path,
            dtype=torch.bfloat16,
        )
        return llm

    def get_response(
        self,
        content: str,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        top_p: float = 1,
    ):
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        response = self.llm.generate(content, sampling_params)
        return response[0].outputs[0].text
