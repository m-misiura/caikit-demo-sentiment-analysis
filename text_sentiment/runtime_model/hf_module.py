import os
from transformers import pipeline
from caikit.core import ModuleBase, ModuleLoader, ModuleSaver, TaskBase, module, task
from text_sentiment.data_model.classification import (
    ClassificationPrediction,
    ClassInfo,
    TextInput,
)


@task(
    required_parameters={"text_input": TextInput},
    output_type=ClassificationPrediction,
)
class HuggingFaceSentimentTask(TaskBase):
    pass


@module(
    "8f72161-c0e4-49b0-8fd0-7587b3017a35",
    "HuggingFaceSentimentModule",
    "0.0.1",
    HuggingFaceSentimentTask,
)
class HuggingFaceSentimentModule(ModuleBase):
    def __init__(self, model_path) -> None:
        super().__init__()
        loader = ModuleLoader(model_path)
        config = loader.config
        model = pipeline(model=config.hf_artifact_path, task="sentiment-analysis")
        self.sentiment_pipeline = model

    def run(self, text_input: TextInput) -> ClassificationPrediction:
        raw_results = self.sentiment_pipeline(text_input.text)
        class_info = []
        for result in raw_results:
            class_info.append(
                ClassInfo(class_name=result["label"], confidence=result["score"])
            )
        return ClassificationPrediction(classes=class_info)

    @staticmethod
    def bootstrap(cls, model_path="distilbert-base-uncased-finetuned-sst-2-english"):
        return cls(model_path=model_path)

    def save(self, model_path, **kwargs):
        module_saver = ModuleSaver(self, model_path=model_path)
        with module_saver:
            rel_path = module_saver.add_dir("hf_moduel")
            save_path = os.path.join(model_path, rel_path)
            self.sentiment_pipeline.save_pretrained(save_path)
            module_saver.add_file({"hf_artifact_path": rel_path})

    @classmethod
    def load(cls, model_path):
        return cls(model_path)
