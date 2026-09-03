"""Vietnamese social-text preprocessing shared by baseline and neural models."""

import re
import unicodedata

from sklearn.base import BaseEstimator, TransformerMixin
from underthesea import word_tokenize

TEEN_CODE = {
    "ko": "không",
    "k": "không",
    "hok": "không",
    "dc": "được",
    "đc": "được",
    "mik": "mình",
    "mk": "mình",
    "mn": "mọi_người",
    "j": "gì",
    "z": "vậy",
    "vs": "với",
    "cx": "cũng",
    "nt": "nhắn_tin",
    "rep": "trả_lời",
}
STOPWORDS = {
    "ạ",
    "à",
    "ấy",
    "bị",
    "bởi",
    "các",
    "cái",
    "cho",
    "của",
    "cùng",
    "đã",
    "đang",
    "đây",
    "để",
    "đến",
    "đó",
    "được",
    "là",
    "lại",
    "mà",
    "một",
    "này",
    "những",
    "ở",
    "rằng",
    "rất",
    "thì",
    "trên",
    "trong",
    "từ",
    "và",
    "với",
}
WORD_RE = re.compile(r"(?u)\b[\wÀ-ỹ]+\b")
EMOJI_RE = re.compile("[\U0001f300-\U0001faff\u2600-\u27bf]")


class SocialPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, texts, y=None):
        return self

    def normalize_one(self, value):
        text = unicodedata.normalize("NFC", str(value)).lower()
        text = re.sub(r"(?:https?://|www\.)\S+", " urltoken ", text)
        text = re.sub(r"(?::|;|=)-?(?:\)|D|\])", " iconpositive ", text)
        text = re.sub(r"(?::|;|=)-?(?:\(|/|\\)", " iconnegative ", text)
        text = EMOJI_RE.sub(" emojitoken ", text)
        text = re.sub(r"([^\W\d_])\1{2,}", r"\1\1", text)
        text = WORD_RE.sub(lambda m: TEEN_CODE.get(m.group(0), m.group(0)), text)
        segmented = word_tokenize(text, format="text")
        return " ".join(token for token in segmented.split() if token not in STOPWORDS)

    def transform(self, texts):
        return [self.normalize_one(text) for text in texts]
