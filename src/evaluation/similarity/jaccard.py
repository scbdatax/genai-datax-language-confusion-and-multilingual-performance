import numpy as np
from typing import List
from transformers import AutoTokenizer
from pythainlp.tokenize import word_tokenize
from huggingface_hub import login
from src.constants import HF_TOKEN


class Jaccard:
    def __init__(self, method: str, model_name: str = None) -> None:
        """
        Initialize instance attributes.

        Args:
            method (str): approach to split a response which is either `simple` or `tokenizer`.
        """
        self.method = method
        
        if (method == "tokenizer") and (model_name is not None):
            self.tokenizer = self._get_tokenizer(model_name)

    @staticmethod
    def _get_tokenizer(model_name: str) -> AutoTokenizer:
        model_path = {
            "typhoon1.5": "scb10x/llama-3-typhoon-v1.5-8b-instruct",
            "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
        }
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path[model_name])
            return tokenizer
        except:
            try:
                login(token=HF_TOKEN)
                tokenizer = AutoTokenizer.from_pretrained(model_path[model_name])
                return tokenizer
            except:
                raise NotImplementedError()
    
    @staticmethod
    def just_split(text: str) -> List[str]:
        """
        Split text with space (only work for English sentences).

        Args:
            text (str): string of an output generated from llm.

        Returns:
            list of words in the text.
        """
        return text.lower().strip().split()
    
    @staticmethod
    def split_with_tokenizer(text: str, tokenizer: AutoTokenizer) -> List[str]:
        """
        Split text with tokenizer.

        Args:
            text (str): string of an output generated from llm.
            tokenizer (AutoTokenizer): model name or model path to download tokenizer.

        Returns:
            list of tokens in the text.
        """
        token_ids = tokenizer.encode(text)
        token_strings = [
            tokenizer.decode(item).strip() 
            for item in token_ids
        ]

        # remove the start token
        token_strings = [
            item 
            for item in token_strings
            if item != "<|begin_of_text|>"
        ]
        return token_strings
    
    @staticmethod
    def split_with_pythai(text: str) -> List[str]:
        """
        Split text with pythai engine.

        Args:
            text (str): string of an output generated from llm.

        Returns:
            list of tokens in the text.
        """
        text = text.lower().strip()
        split_text_list = [
            t
            for t in word_tokenize(text, engine="newmm")
            if t != " "
        ]
        return split_text_list
    
    def split_text(
        self,
        text: str,
    ) -> List[str]:
        """
        Wrapper for text splitter. Currently support splitting with white spaces and tokenizer.

        Args:
            text (str): string of an output generated from llm.
            model_name (str): model name or model path to download tokenizer.

        Returns:
            list of words/tokens in the text.
        """
        if self.method == "simple":
            return self.just_split(text)
        elif self.method == "tokenizer":
            return self.split_with_tokenizer(text, self.tokenizer)
        elif self.method == "pythai":
            return self.split_with_pythai(text)
        else:
            raise NotImplementedError()
        
    def prepare_split_responses(self, all_responses: List[str]) -> List[List[str]]:
        """
        Split text of all responses and store as a list.

        Args:
            all_responses (list): list of N original respones.

        Returns:
            list of split responses.
        """
        all_split_responses = [
            set(self.split_text(res)) 
            for res in all_responses
        ]
        return all_split_responses

    @staticmethod
    def cal_jaccard_sim(all_split_responses: List[List[str]]) -> np.array:
        """
        Calculate the weighted adjacency matrix (W) using the Jaccard simmilarity.

        Args: 
            all_split_responses (list): list of N split responses.

        Returns:
            similarity matrix from jaccard similarity.
        """
        ret = np.eye(len(all_split_responses))
        for i, res_i in enumerate(all_split_responses):
            for j, res_j in enumerate(all_split_responses[i+1:], i+1):
                ret[i,j] = ret[j,i] = len(res_i.intersection(res_j)) / max(len(res_i.union(res_j)), 1)
        return ret

    def get_W_mat(
        self,
        all_responses: List[str],
    ) -> np.array:
        """
        Args:
            all_responses (list): list of N original respones.

        Returns:
            weighted adjacency matrix (W).
        """
        all_split_responses = self.prepare_split_responses(all_responses)
        W = self.cal_jaccard_sim(all_split_responses)
        return W
