from typing import List

from caikit.core import DataObjectBase
from caikit.core.data_model import dataobject


@dataobject(package="text_sentiment.data_model")
class ClassInfo(DataObjectBase):
    class_name: str
    confidence: float


@dataobject(package="text_sentiment.data_model")
class ClassificationPrediction(DataObjectBase):
    classes: List[ClassInfo]


@dataobject(package="text_sentiment.data_model")
class TextInput(DataObjectBase):
    text: str
