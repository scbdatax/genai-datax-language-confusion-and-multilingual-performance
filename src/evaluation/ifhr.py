from typing import Tuple
from src.evaluation import LanguageID
from src.evaluation.similarity import Jaccard


class SummarizationIFHR:
    def __init__(self) -> None:
        pass

    @staticmethod
    def identify_lang(text: str) -> str:
        """
        Identify the language of the text.

        Args:
            text (str): response from an llm.

        Returns:
            detected language which is supposed to be either `en` or `th`.
        """
        text = text.replace("\n", " ")
        lid = LanguageID()
        output = lid.predict(text)
        return output[0]
    
    @staticmethod
    def count_n_words(text: str, lang: str) -> int:
        """
        Count the number of words in the text. 
        If it is written in English, count the number of words by splitting with spaces.
        If it is written in Thai, count the number of words by tokenizing text with PyThaiNLP.

        Args:
            text (str): response from an llm.
            lang (str): expected langauge.

        Returns:
            number of words in the text.
        """
        text = text.strip()
        if lang == "en":
            n_words = len(text.split())

        elif lang == "th":
            tokenizer = Jaccard(method="pythai")
            n_words = len(tokenizer.split_with_pythai(text))

        return n_words
    
    @staticmethod
    def count_n_paragraphs(text: str) -> int:
        """
        Count the number of paragraphs using new line character.

        Args:
            text (str): response from an llm.

        Returns:
            number of paragraphs in the text.
        """
        text = text.strip()
        return len(text.split("\n"))
    
    def check_ifhr_by_record(
            self, 
            text: str, 
            lang: str,
            return_scores: bool = True,
        ) -> int | Tuple:
        """
        Assign 0 if the text satisfies the following criteria
        - output language should be written in the expected language.
        - number of words >= 100.
        - number of paragraphs = 1.
        Otherwise, assign 1.

        Args:
            text (str): response from an llm.
            lang (str): expected langauge.
            return_scores (bool): boolean controlling the output of this function.

        Returns:    
            either just the flag, or the flag together with other scores used in the criteria.
        """
        lang_detected = self.identify_lang(text)
        n_words = self.count_n_words(text, lang)
        n_paragraphs = self.count_n_paragraphs(text)

        if (lang_detected == lang) and (n_words >= 100) and n_paragraphs == 1:
            flag = 0
        else:
            flag = 1

        if return_scores:
            return (flag, lang_detected, n_words, n_paragraphs)
        else:
            return flag
        