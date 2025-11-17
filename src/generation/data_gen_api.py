import os
import logging
import pandas as pd
from tqdm import tqdm
from src.llm.api import LLM
from datasets import load_dataset
from tiktoken import encoding_for_model


class MMLU:
    def __init__(
        self,
        split: str = "test",
        lang: str = "en",
    ) -> None:

        self.ds = load_dataset("cais/mmlu", "all")[split]
        self.lang = lang
        
        if lang == "en":
            self.prompt_template = "The following is a multiple choices question about {}. Please choose the most correct choice. The answer should be one among [A, B, C, D]. Don't include any explanation. Generate only answer.\nQuestion : {}\nA. {}\nB. {}\nC. {}\nD. {}\nAnswer : "
        elif lang == "th":
            self.prompt_template = "ต่อไปนี้เป็นคำถามแบบหลายตัวเลือกเกี่ยวกับ {}. กรุณาเลือกคำตอบที่ถูกต้องที่สุด คำตอบควรเป็นหนึ่งใน [A, B, C, D]. ไม่ต้องเพิ่มคำอธิบาย ให้ตอบแต่คำตอบ\nคำถาม : {}\nA. {}\nB. {}\nC. {}\nD. {}\nคำตอบ : "
        else:
            raise Exception("Please choose the language to be either 'en' or 'th'")
        
        self.answer_map_dict = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D'
        }

        self.pred_df = {
            'ID': [],
            'answer_pred' : []
        }
    
    def generate_questions(
        self,
    ) -> None:
        
        df = self.ds.to_pandas()
        df['ID'] = df.index
        df = df.replace({'answer': self.answer_map_dict})
        df.rename(
            columns={
                "answer": "answer_gt",
            }
        )
        df.to_json("generated_data/MMLU/MMLU_questions_gt_en.jsonl")


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
        
        # logging config
        if os.path.exists(f"data_gen_MMLU_temp_{temperature}.log"):
            os.remove(f"data_gen_MMLU_temp_{temperature}.log")

        logging.basicConfig(
            filename=f"data_gen_MMLU_temp_{temperature}.log",
            filemode='a',
            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S',
            level=logging.INFO
        )
        
        self.logger = logging.getLogger(__name__)

        # generate responses
        question_id = resume_id
        self.ds = self.ds.filter(lambda example, idx: idx >= resume_id, with_indices=True)

        for rows in tqdm(self.ds):
            question = rows['question']
            subject = rows['subject']
            choices = rows['choices']
            prompt = self.prompt_template.format(
                        subject, 
                        question, 
                        choices[0], 
                        choices[1], 
                        choices[2], 
                        choices[3]
                    )

            for _ in range(num_gen):
                res = llm.get_response(
                            prompt,
                            temperature=temperature, 
                            top_p=top_p, 
                            max_tokens=max_tokens
                        )
                
                self.pred_df['ID'].append(question_id)
                self.pred_df['answer_pred'].append(res)
                pd.DataFrame(self.pred_df).to_json(f'{llm_name}_MMLU_temp_{temperature}_{self.lang}-it_answers.jsonl')
            
            # log
            self.logger.info(f"Finished question_id : {question_id}")
            question_id += 1


class WTI_MC:
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
            question_gt_path = "generated_data/MMLU/MMLU_questions_gt_en.jsonl"
        else:
            raise Exception("Please choose the language to be 'en' only")

        self.ds = load_dataset("csv", data_files=question_gt_path)["train"]

        self.pred_df = {
            'ID': [],
            'answer_pred' : []
        }

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
        
        # logging config
        if os.path.exists(f"data_gen_WTI_MC_temp_{temperature}_{self.inst_lang}-it_{self.input_lang}-input.log"):
            os.remove(f"data_gen_WTI_MC_temp_{temperature}_{self.inst_lang}-it_{self.input_lang}-input.log")

        logging.basicConfig(
            filename=f"data_gen_WTI_MC_temp_{temperature}_{self.inst_lang}-it_{self.input_lang}-input.log",
            filemode='a',
            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S',
            level=logging.INFO
        )
        
        self.logger = logging.getLogger(__name__)

        # generate responses
        question_id = resume_id
        self.ds = self.ds.filter(lambda example, idx: idx >= resume_id, with_indices=True)

        for rows in tqdm(self.ds):
            id = rows['ID']
            question = rows['question']
            subject = rows['subject']
            prompt = self.prompt_template.format(
                        subject, 
                        question
                    )

            for _ in range(num_gen):
                res = llm.get_response(
                            prompt,
                            temperature=temperature, 
                            top_p=top_p, 
                            max_tokens=max_tokens
                        )
                
                self.pred_df['ID'].append(id)
                self.pred_df['answer_pred'].append(res)

                # save data
                pd.DataFrame(self.pred_df).to_json(
                    f'{llm_name}_WTI_MC_temp_{temperature}_{self.inst_lang}-it_{self.input_lang}-input_answers.jsonl', 
                )
            
            # log
            self.logger.info(f"Finished question_id : {question_id}")
            question_id += 1


class WTI_CQA:
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

        try:
            question_gt_path = f"generated_data/WTI_CQA/WTI_CQA_questions_gt_{input_lang}.jsonl"
        except:
            raise Exception("Please choose the language to be either 'en' or 'th'")

        self.ds = load_dataset("csv", data_files=question_gt_path)["train"]

        self.pred_df = {
            'ID': [],
            'answer_pred' : []
        }
    
    @staticmethod
    def _calculate_tokens(text: str, model: str = "gpt-4o") -> int:
        encoding = encoding_for_model(model)
        return len(encoding.encode(text))

    def check_lengthy_prompt(
        self,
        text: str, 
        max_tokens: int = 8_192,
        model: str = "gpt-4o",
    ) -> bool:
        """
        Check if a prompt exceeds maximum tokens.
        """
        return self._calculate_tokens(text, model) > max_tokens

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
        
        # logging config
        if os.path.exists(f"data_gen_WTI_CQA_temp_{temperature}_{self.inst_lang}-it_{self.input_lang}-in_{self.output_lang}-out.log"):
            os.remove(f"data_gen_WTI_CQA_temp_{temperature}_{self.inst_lang}-it_{self.input_lang}-in_{self.output_lang}-out.log")

        logging.basicConfig(
            filename=f"data_gen_WTI_CQA_temp_{temperature}_{self.inst_lang}-it_{self.input_lang}-in_{self.output_lang}-out.log",
            filemode='a',
            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S',
            level=logging.INFO
        )
        
        self.logger = logging.getLogger(__name__)

        # generate responses
        question_id = resume_id
        self.ds = self.ds.filter(lambda example, idx: idx >= resume_id, with_indices=True)

        for rows in tqdm(self.ds):
            id = rows['ID']
            question = rows['question']
            context = rows['context']
            subject = rows['subject']
            prompt = self.prompt_template.format(
                        subject, 
                        self.output_lang_inst,
                        context,
                        question
                    )
            
            is_lengthy = self.check_lengthy_prompt(prompt, max_tokens=int(6_000))

            if is_lengthy:
                self.logger.info(f"Skipped question_id : {question_id} (Lengthy)")
                continue

            for _ in range(num_gen):
                res = llm.get_response(
                            prompt,
                            temperature=temperature, 
                            top_p=top_p, 
                            max_tokens=max_tokens
                        )
                
                self.pred_df['ID'].append(id)
                self.pred_df['answer_pred'].append(res)

                # save data
                pd.DataFrame(self.pred_df).to_json(
                    f'{llm_name}_WTI_CQA_temp_{temperature}_{self.inst_lang}-it_{self.input_lang}-in_{self.output_lang}-out_answers.jsonl', 
                )
            
            # log
            self.logger.info(f"Finished question_id : {question_id}")
            question_id += 1


class WTI_SUM:
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

        try:
            question_gt_path = f"generated_data/WTI_SUM/WTI_SUM_questions_gt_{input_lang}.jsonl"
        except:
            raise Exception("Please choose the language to be either 'en' or 'th'")

        self.ds = load_dataset("csv", data_files=question_gt_path)["train"]

        self.pred_df = {
            'ID': [],
            'answer_pred' : []
        }
    
    @staticmethod
    def _calculate_tokens(text: str, model: str = "gpt-4o") -> int:
        encoding = encoding_for_model(model)
        return len(encoding.encode(text))

    def check_lengthy_prompt(
        self,
        text: str, 
        max_tokens: int = 8_192,
        model: str = "gpt-4o",
    ) -> bool:
        """
        Check if a prompt exceeds maximum tokens.
        """
        return self._calculate_tokens(text, model) > max_tokens

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
        
        # logging config
        if os.path.exists(f"data_gen_WTI_SUM_temp_{temperature}_{self.inst_lang}-it_{self.input_lang}-in_{self.output_lang}-out.log"):
            os.remove(f"data_gen_WTI_SUM_temp_{temperature}_{self.inst_lang}-it_{self.input_lang}-in_{self.output_lang}-out.log")

        logging.basicConfig(
            filename=f"data_gen_WTI_SUM_temp_{temperature}_{self.inst_lang}-it_{self.input_lang}-in_{self.output_lang}-out.log",
            filemode='a',
            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S',
            level=logging.INFO
        )
        
        self.logger = logging.getLogger(__name__)

        # generate responses
        question_id = resume_id
        self.ds = self.ds.filter(lambda example, idx: idx >= resume_id, with_indices=True)

        for rows in tqdm(self.ds):
            id = rows['ID']
            context = rows['context']
            question = rows['question']

            prompt = self.prompt_template.format(
                question=question,
                lang=self.output_lang_inst,
                context=context,
            )
            
            is_lengthy = self.check_lengthy_prompt(prompt, max_tokens=int(6_000))

            if is_lengthy:
                self.logger.info(f"Skipped question_id : {question_id} (Lengthy)")
                continue

            for _ in range(num_gen):
                res = llm.get_response(
                    prompt,
                    temperature=temperature, 
                    top_p=top_p, 
                    max_tokens=max_tokens
                )
                
                self.pred_df['ID'].append(id)
                self.pred_df['answer_pred'].append(res)

                # save data
                pd.DataFrame(self.pred_df).to_json(
                    f'{llm_name}_WTI_SUM_temp_{temperature}_{self.inst_lang}-it_{self.input_lang}-in_{self.output_lang}-out_answers.jsonl', 
                )
            
            # log
            self.logger.info(f"Finished question_id : {question_id}")
            question_id += 1


class THAI_EXAM:
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

        try:
            question_gt_path = f"generated_data/THAI_EXAM/THAI_EXAM_questions_gt_{input_lang}.jsonl"
        except:
            raise Exception("Please choose the language to be either 'en' or 'th'")

        self.ds = load_dataset("csv", data_files=question_gt_path)["train"]

        self.pred_df = {
            'ID': [],
            'answer_pred' : []
        }

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
        
        # logging config
        if os.path.exists(f"data_gen_THAI_EXAM_temp_{temperature}_{self.inst_lang}-it_{self.input_lang}-input.log"):
            os.remove(f"data_gen_THAI_EXAM_temp_{temperature}_{self.inst_lang}-it_{self.input_lang}-input.log")

        logging.basicConfig(
            filename=f"data_gen_THAI_EXAM_temp_{temperature}_{self.inst_lang}-it_{self.input_lang}-input.log",
            filemode='a',
            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S',
            level=logging.INFO
        )
        
        self.logger = logging.getLogger(__name__)

        # generate responses
        question_id = resume_id
        self.ds = self.ds.filter(lambda example, idx: idx >= resume_id, with_indices=True)

        for rows in tqdm(self.ds):
            id = rows['ID']
            question = rows['question']
            prompt = self.prompt_template.format(question)

            for _ in range(num_gen):
                res = llm.get_response(
                    prompt,
                    temperature=temperature, 
                    top_p=top_p, 
                    max_tokens=max_tokens
                )
                
                self.pred_df['ID'].append(id)
                self.pred_df['answer_pred'].append(res)

                # save data
                pd.DataFrame(self.pred_df).to_json(
                    f'{llm_name}_THAI_EXAM_temp_{temperature}_{self.inst_lang}-it_{self.input_lang}-input_answers.jsonl', 
                )
            
            # log
            self.logger.info(f"Finished question_id : {question_id}")
            question_id += 1
