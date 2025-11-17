import re
import jieba
import string
import fasttext
from typing import List, Tuple
from pythainlp.tokenize import word_tokenize

class LanguageID:
    def __init__(self) -> None:
        pretrained_lang_model = "/tmp/lid.176.bin"
        self.model = fasttext.load_model(pretrained_lang_model)
        self.punctuations = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."

    def predict(
            self, 
            text: str,
            k: int = 1,
        ) -> List[str]:
        """
        Identify the language of the text using fasttext library.

        Args:
            text (str): response from an llm.
            k (int): number of tokens that

        Returns:
            detected language which is supposed to be either `en` or `th`.
        """

        predictions = self.model.predict(text, k=k)[0]
        langs = []
        for p in predictions:
            langs.append(p.replace('__label__', ''))

        return langs

    def predict_with_prob(
            self, 
            text: str,
            k: int = 1,
        ) -> Tuple[List[str], List[float]]:
        """
        Identify the language of the text using fasttext library.

        Args:
            text (str): response from an llm.
            k (int): number of tokens that

        Returns:
            Tuple of detected language and probability of confidence
        """

        predictions = self.model.predict(text, k=k)
        langs = []
        probs = predictions[1]
        for p in predictions[0]:
            langs.append(p.replace('__label__', ''))

        return langs, probs

    def pred_word_level(
            self, 
            text: str,
            prob_ratio_thresh: float = 10,
        ) -> List[str]:
        """
        Identify the language of the text in word-level using fasttext library.

        Args:
            text (str): response from an llm.
            prob_ratio_thresh (float): we will omit the predicted lang if the ratio of prob 
            of the first language to the second one is less than a certain value of threshold.

        Returns:
            A list of detected language code of each word.
        """

        word_level_langs = []

        cleaned_text = re.sub(r'\d', '', text)
        cleaned_text = re.sub(r"[%s]+" %self.punctuations, "", cleaned_text)
        cleaned_text = ''.join([w for w in cleaned_text if w not in string.punctuation])

        tokens = word_tokenize(cleaned_text)
        for t in tokens:
            if re.search(r'\s+', t):
                continue
            else:
                langs, probs = self.predict_with_prob(t, k=2)
                lang = langs[0]
                if (len(probs) == 1) or (probs[0] / probs[1] > prob_ratio_thresh):
                    if lang == 'zh':
                        for zht in list(jieba.cut(t)):
                            word_level_langs.append(self.predict(zht)[0])
                    else:
                        word_level_langs.append(lang)
        return word_level_langs