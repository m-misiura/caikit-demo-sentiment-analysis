import os
import grpc
from caikit.runtime.service_factory import ServicePackageFactory
from text_sentiment.data_model import TextInput

inference_service = ServicePackageFactory().get_service_package(
    ServicePackageFactory.ServiceType.INFERENCE
)
port = 8085
channel = grpc.insecure_channel(f"localhost:{port}")
client_stub = inference_service.stub_class(channel)

for text in ["I love this!", "I hate this!"]:
    input_text_proto = TextInput(text=text).to_proto()
    request = inference_service.messages.HuggingFaceSentimentTaskRequest(
        text_input=input_text_proto
    )
    response = client_stub.HuggingFaceSentimentTaskPredict(
        request, metadata=[("mm-model-id", "text_sentiment")]
    )
    print("Text", text)
    print("Response", response)
