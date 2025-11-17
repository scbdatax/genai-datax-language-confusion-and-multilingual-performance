from src.llm.api import LLM
from datasets import Dataset
from datasets import load_dataset
from datasets import concatenate_datasets
from tiktoken import encoding_for_model


class BaseGenerator:
    @staticmethod
    def _calculate_tokens(text: str, model: str = "gpt-4o") -> int:
        encoding = encoding_for_model(model)
        return len(encoding.encode(text))

    def _check_lengthy_prompt(
        self,
        text: str, 
        max_tokens: int = 8_192,
        model: str = "gpt-4o",
    ) -> bool:
        """
        Check if a prompt exceeds maximum tokens.
        """
        return self._calculate_tokens(text, model) > max_tokens

    def _prepare_dataset(
        self, 
        ds: Dataset, 
        num_gen: int = 10,
        resume_id: int = 0,
    ) -> Dataset:
        # check and filter out if it's lengthy
        ds = ds.map(
            lambda x: {
                "is_lengthy": self._check_lengthy_prompt(x["prompt"], max_tokens=int(6_000))
            }
        )
        ds = ds.filter(lambda x: ~x["is_lengthy"])

        # resume the last checkpoint
        ds = ds.filter(lambda example, idx: idx >= resume_id, with_indices=True)

        # duplicate by `num_gen` times for each query
        ds = concatenate_datasets([self.ds] * num_gen)
        ds = ds.sort("ID")
        return ds


class WTI_CQA(BaseGenerator):
    def __init__(
        self,
        inst_lang: str = "th",
        input_lang: str = "th",
        output_lang: str = "th",
    ) -> None:
        self.inst_lang = inst_lang
        self.input_lang = input_lang
        self.output_lang = output_lang

        if inst_lang == "en":
            if self.output_lang == "en":
                self.output_lang_inst = "English"
            elif self.output_lang == "th":
                self.output_lang_inst = "Thai"
            else:
                raise Exception("Please choose the language to be either 'en' or 'th'")
            self.prompt_template = "The following is a question with context about {}. Please answer according to the context in {}. \nContext : \n{} \nQuestion : {}\nAnswer : "
        elif inst_lang == "th":
            if self.output_lang == "en":
                self.output_lang_inst = "อังกฤษ"
            elif self.output_lang == "th":
                self.output_lang_inst = "ไทย"
            else:
                raise Exception("Please choose the language to be either 'en' or 'th'")
            self.prompt_template = "ต่อไปนี้เป็นคำถามเกี่ยวกับ {} กรุณาตอบโดยอ้างอิงจากบทความที่ให้มา ให้ตอบเป็นภาษา {} \nบทความ : \n{} \nคำถาม : {}\nคำตอบ : "
        else:
            raise Exception("Please choose the language to be either 'en' or 'th'")

        if input_lang == "en":
            question_gt_path = "generated_data/WTI_CQA/WTI_CQA_questions_gt_en.csv"
        elif input_lang == "th":
            question_gt_path = "generated_data/WTI_CQA/WTI_CQA_questions_gt.csv"
        else:
            raise Exception("Please choose the language to be either 'en' or 'th'")

        self.ds = load_dataset("csv", data_files=question_gt_path)["train"]

    def generate_responses(
        self,
        llm: LLM,
        llm_name: str,
        num_gen: int = 10,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 1000,
        resume_id: int = 0,
    ) -> None:
        # prepare prompt
        self.ds = self.ds.map(
            lambda x: {
                "prompt": self.prompt_template.format(
                    x["subject"], 
                    self.output_lang_inst,
                    x["context"],
                    x["question"],
                )
            }
        )
        self.ds = self._prepare_dataset(self.ds, num_gen, resume_id)

        # generate responses
        self.ds = self.ds.map(
            lambda x: {
                "answer_pred": llm.get_response(
                    x["prompt"],
                    temperature=temperature, 
                    top_p=top_p, 
                    max_tokens=max_tokens,
                )
            }
        )

        filename = f"{llm_name}_WTI_CQA_temp_{temperature}_{self.inst_lang}-it_{self.input_lang}-in_{self.output_lang}-out_answers.jsonl"
        (
            self.ds
            .select_columns(["ID", "answer_pred"])
            .to_json(filename)
        )


class WTI_SUM(BaseGenerator):
    def __init__(
        self,
        inst_lang: str = "th",
        input_lang: str = "th",
        output_lang: str = "th",
    ) -> None:
        self.inst_lang = inst_lang
        self.input_lang = input_lang
        self.output_lang = output_lang

        if inst_lang == "en":
            if self.output_lang == "en":
                self.output_lang_inst = "an English"
            elif self.output_lang == "th":
                self.output_lang_inst = "a Thai"
            else:
                raise Exception("Please choose the language to be either 'en' or 'th'")
            self.prompt_template = "{question}\nReturn a response in a format of {lang} paragraph with at least 100 words.\n\nArticle: {context}\n\nSummary: "
        elif inst_lang == "th":
            if self.output_lang == "en":
                self.output_lang_inst = "อังกฤษ"
            elif self.output_lang == "th":
                self.output_lang_inst = "ไทย"
            else:
                raise Exception("Please choose the language to be either 'en' or 'th'")
            self.prompt_template = "{question}\nให้ตอบเป็นภาษา{lang}ใน 1 ย่อหน้าโดยใช้จำนวนคำอย่างต่ำ 100 คำ\n\nบทความ: {context}\n\nสรุป: "
        else:
            raise Exception("Please choose the language to be either 'en' or 'th'")

        if input_lang == "en":
            question_gt_path = "generated_data/WTI_SUM/WTI_SUM_questions_gt_en.csv"
        elif input_lang == "th":
            question_gt_path = "generated_data/WTI_SUM/WTI_SUM_questions_gt.csv"
        else:
            raise Exception("Please choose the language to be either 'en' or 'th'")

        self.ds = load_dataset("csv", data_files=question_gt_path)["train"]

    def generate_responses(
        self,
        llm: LLM,
        llm_name: str,
        num_gen: int = 10,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 1000,
        resume_id: int = 0,
    ) -> None:
        # prepare prompt
        self.ds = self.ds.map(
            lambda x: {
                "prompt": self.prompt_template.format(
                    question=x["question"],
                    lang=self.output_lang_inst,
                    context=x["context"],
                )
            }
        )
        self.ds = self._prepare_dataset(self.ds, num_gen, resume_id)

        # generate responses
        self.ds = self.ds.map(
            lambda x: {
                "answer_pred": llm.get_response(
                    x["prompt"],
                    temperature=temperature, 
                    top_p=top_p, 
                    max_tokens=max_tokens,
                )
            }
        )

        filename = f"{llm_name}_WTI_SUM_temp_{temperature}_{self.inst_lang}-it_{self.input_lang}-in_{self.output_lang}-out_answers.jsonl"
        (
            self.ds
            .select_columns(["ID", "answer_pred"])
            .to_json(filename)
        )


class MMLU(BaseGenerator):
    def __init__(
        self,
        split: str = "dev", # change split to "test" for the full test
        lang: str = "en"
    ) -> None:

        self.ds = load_dataset("cais/mmlu", "all")[split]
        self.lang = lang
        
        if lang == "en":
            self.prompt_template = "The following is a multiple choices question about {}. Please choose the most correct choice. The answer should be one among [A, B, C, D]. Don't include any explanation. Generate only answer.\nQuestion : {}\nA. {}\nB. {}\nC. {}\nD. {}\nAnswer : "
        elif lang == "th":
            self.prompt_template = "ต่อไปนี้เป็นคำถามแบบหลายตัวเลือกเกี่ยวกับ {}. กรุณาเลือกคำตอบที่ถูกต้องที่สุด คำตอบควรเป็นหนึ่งใน [A, B, C, D]. ไม่ต้องเพิ่มคำอธิบาย ให้ตอบแต่คำตอบ\nคำถาม : {}\nA. {}\nB. {}\nC. {}\nD. {}\nคำตอบ : "
        else:
            raise Exception("Please choose the language to be either 'en' or 'th'")

    def generate_responses(
        self,
        llm: LLM,
        llm_name: str,
        num_gen: int = 10,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 1,
        resume_id: int = 0,
    ) -> None:
        # prepare prompt
        self.ds = self.ds.map(
            lambda x: {
                "prompt": self.prompt_template.format(
                    x["question"],
                    x["subject"],
                    x["choices"][0],
                    x["choices"][1],
                    x["choices"][2],
                    x["choices"][3],
                )
            }
        )
        self.ds = self.ds.map(
            lambda x, idx: {"ID": idx}, 
            with_indices=True,
        )
        self.ds = self._prepare_dataset(self.ds, num_gen, resume_id)

        # generate responses
        self.ds = self.ds.map(
            lambda x: {
                "answer_pred": llm.get_response(
                    x["prompt"],
                    temperature=temperature, 
                    top_p=top_p, 
                    max_tokens=max_tokens,
                )
            }
        )

        filename = f"{llm_name}_MMLU_temp_{temperature}_{self.lang}-it_answers.jsonl"
        (
            self.ds
            .select_columns(["ID", "answer_pred"])
            .to_json(filename)
        )


class WTI_MC(BaseGenerator):
    def __init__(
        self,
        inst_lang: str = "th",
        input_lang: str = "th",
    ) -> None:
        self.inst_lang = inst_lang
        self.input_lang = input_lang

        if inst_lang == "en":
            self.prompt_template = "The following is a multiple choices question about {}. Please choose the most correct choice. The answer should be one among [A, B, C, D]. Don't include any explanation. Generate only answer.\nQuestion : {}\nAnswer : "
        elif inst_lang == "th":
            self.prompt_template = "ต่อไปนี้เป็นคำถามแบบหลายตัวเลือกเกี่ยวกับ {}. กรุณาเลือกคำตอบที่ถูกต้องที่สุด คำตอบควรเป็นหนึ่งใน [A, B, C, D]. ไม่ต้องเพิ่มคำอธิบาย ให้ตอบแต่คำตอบ\nคำถาม : {}\nคำตอบ : "
        else:
            raise Exception("Please choose the language to be either 'en' or 'th'")

        if input_lang == "en":
            question_gt_path = "generated_data/WTI_MC/WTI_MC_questions_gt_en_cleaned.csv"
        elif input_lang == "th":
            question_gt_path = "generated_data/WTI_MC/WTI_MC_questions_gt_cleaned.csv"
        else:
            raise Exception("Please choose the language to be either 'en' or 'th'")

        self.ds = load_dataset("csv", data_files=question_gt_path)["train"]

    def generate_responses(
        self,
        llm: LLM,
        llm_name: str,
        num_gen: int = 10,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 10,
        resume_id: int = 0,
    ) -> None:
        # prepare prompt
        self.ds = self.ds.map(
            lambda x: {
                "prompt": self.prompt_template.format(
                    x["question"],
                    x["subject"],
                )
            }
        )
        self.ds = self._prepare_dataset(self.ds, num_gen, resume_id)

        # generate responses
        self.ds = self.ds.map(
            lambda x: {
                "answer_pred": llm.get_response(
                    x["prompt"],
                    temperature=temperature, 
                    top_p=top_p, 
                    max_tokens=max_tokens,
                )
            }
        )

        filename = f"{llm_name}_WTI_MC_temp_{temperature}_{self.inst_lang}-it_{self.input_lang}-input_answers.jsonl"
        (
            self.ds
            .select_columns(["ID", "answer_pred"])
            .to_json(filename)
        )


class THAI_EXAM(BaseGenerator):
    def __init__(
        self,
        inst_lang: str = "th",
        input_lang: str = "th",
    ) -> None:
        self.inst_lang = inst_lang
        self.input_lang = input_lang

        if inst_lang == "en":
            self.prompt_template = "Please choose the most correct choice. The answer should be one among [A, B, C, D, E]. Don't include any explanation. Generate only answer.\nQuestion : {}\nAnswer : "
        elif inst_lang == "th":
            self.prompt_template = "กรุณาเลือกคำตอบที่ถูกต้องที่สุด คำตอบควรเป็นหนึ่งใน [A, B, C, D, E]. ไม่ต้องเพิ่มคำอธิบาย ให้ตอบแต่คำตอบ\nคำถาม : {}\nคำตอบ : "
        else:
            raise Exception("Please choose the language to be either 'en' or 'th'")

        if input_lang == "en":
            question_gt_path = "generated_data/THAI_EXAM/THAI_EXAM_questions_gt_en.csv"
        elif input_lang == "th":
            question_gt_path = "generated_data/THAI_EXAM/THAI_EXAM_questions_gt.csv"
        else:
            raise Exception("Please choose the language to be either 'en' or 'th'")

        self.ds = load_dataset("csv", data_files=question_gt_path)["train"]

    def generate_responses(
        self,
        llm: LLM,
        llm_name: str,
        num_gen: int = 10,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 10,
        resume_id: int = 0,
    ) -> None:
        # prepare prompt
        self.ds = self.ds.map(
            lambda x: {
                "prompt": self.prompt_template.format(x["question"])
            }
        )
        self.ds = self._prepare_dataset(self.ds, num_gen, resume_id)

        # generate responses
        self.ds = self.ds.map(
            lambda x: {
                "answer_pred": llm.get_response(
                    x["prompt"],
                    temperature=temperature, 
                    top_p=top_p, 
                    max_tokens=max_tokens,
                )
            }
        )

        filename = f"{llm_name}_THAI_EXAM_temp_{temperature}_{self.inst_lang}-it_{self.input_lang}-input_answers.jsonl"
        (
            self.ds
            .select_columns(["ID", "answer_pred"])
            .to_json(filename)
        )
