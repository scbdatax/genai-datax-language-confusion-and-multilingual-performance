import torch
import numpy as np
from typing import List
from typing import Dict
from typing import Union
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.nn.functional import softmax


class NLI:
    def __init__(self, model_name: str = "microsoft/deberta-large-mnli") -> None:
        """
        Initialize instance attributes.

        Args:
            model_name (str): NLI model path.
        """
        self.model =  AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.model.device

    @torch.no_grad()
    def _batch_pred(
        self, 
        sen_1: List[str], 
        sen_2: List[str], 
        max_batch_size: int = 128
    ) -> torch.Tensor:
        """
        Determine whether the two sentences are either contradict, neutral, or entail from NLI model.
        
        Args:
            sen_1 (list): list of responses.
            sen_2 (list): list of responses.
            max_batch_size (int): maximum size of input to predict at a time.

        Returns:
            tensor representing logits of each sentencee pair, noting that (0, 1, 2) = (contradict, neutral, entail).
        """
        inputs = [
            _[0] + " [SEP] " + _[1] 
            for _ in zip(sen_1, sen_2)
        ]

        inputs = self.tokenizer(inputs, padding=True, truncation=True)
        input_ids = torch.tensor(inputs["input_ids"]).to(self.device)
        attention_mask = torch.tensor(inputs["attention_mask"]).to(self.device)

        logits = []
        for st in range(0, len(input_ids), max_batch_size):
            ed = min(st + max_batch_size, len(input_ids))
            prediction = self.model(
                input_ids=input_ids[st:ed],
                attention_mask=attention_mask[st:ed]
            )
            logits.append(prediction["logits"])

        return torch.cat(logits, dim=0)

    @torch.no_grad()
    def create_sim_mat_batched(
        self, 
        question: str, 
        answers: List[str],
    ) -> Dict[str, Union[List[int], torch.Tensor]]:
        """
        Calculate logits from NLI.

        Args:
            question (str): input query.
            answers (list): list of responses.

        Returns:
            Mapping between unique respones and their indices, and logits from the NLI.
        """
        unique_ans = sorted(list(set(answers)))
        semantic_set_ids = {
            ans: i 
            for i, ans in enumerate(unique_ans)
        }
        _rev_mapping = semantic_set_ids.copy()
        sim_mat_batch = torch.zeros(
            (len(unique_ans), len(unique_ans), 3)
        )
        anss_1, anss_2, indices = [], [], []
        for i, ans_i in enumerate(unique_ans):
            for j, ans_j in enumerate(unique_ans):
                if i == j: continue
                anss_1.append(f"{question} {ans_i}")
                anss_2.append(f"{question} {ans_j}")
                indices.append((i,j))
        if len(indices) > 0:
            sim_mat_batch_flat = self._batch_pred(anss_1, anss_2)
            for _, (i,j) in enumerate(indices):
                sim_mat_batch[i,j] = sim_mat_batch_flat[_]
        return {
            "mapping": [_rev_mapping[_] for _ in answers],
            "sim_mat": sim_mat_batch,
        }

    @staticmethod
    @torch.no_grad()
    def _get_sim_mat(
        sims: Dict[str, Union[List[int], torch.Tensor]],
        prob_option: str,
    ) -> np.array:
        """
        Retrieve probability from the logits and construct as a similarity matrix.

        Args:
            sims (dict): dictionary of the mapping and logits.
            prob_option (str): option to use a probability which is either `entail` or `contra`.

        Return:
            similarity matrix (to use to construct W).
        """
        sim_mat = sims["sim_mat"].clone()
        # adjust a pair of identical sentences
        sim_mat[
            torch.arange(sim_mat.shape[0]), torch.arange(sim_mat.shape[0]), :
        ] = torch.tensor([-torch.inf, -torch.inf, 100])
        sim_mat = softmax(sim_mat, dim=2)
        mapping = sims["mapping"]

        n = len(mapping)
        ret = np.ones((n, n))

        if prob_option == "entail":
            for i in range(n):
                for j in range(n):
                    ret[i,j] = sim_mat[mapping[i], mapping[j], 2].item()
        elif prob_option == "contra":
            for i in range(n):
                for j in range(n):
                    ret[i,j] = 1 - sim_mat[mapping[i], mapping[j], 0].item()
        else:
            raise NotImplementedError()
        
        return ret
    
    @staticmethod
    def _get_W_mat(ret: np.array) -> np.array:
        """
        Get the weighted adjacency matrix

        Args:
            ret (np.array): similar matrix.

        Returns:
            weighted adjacency matrix (W).
        """
        n = ret.shape[0]
        W = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                W[i,j] = ( ret[i, j] + ret[j, i] ) / 2
        return W

    def get_W_mat(
        self, 
        question: str, 
        answers: List[str],
        prob_option: str = "entail",
    ) -> np.array:
        """
        Main function to compute the weighted adjacency matrix.

        Args:
            question (str): input query.
            answers (list): list of responses.
            prob_option (str): option to use a probability which is either `entail` or `contra`.
        """
        sims = self.create_sim_mat_batched(question, answers)
        ret = self._get_sim_mat(sims, prob_option)
        W = self._get_W_mat(ret)
        return W
