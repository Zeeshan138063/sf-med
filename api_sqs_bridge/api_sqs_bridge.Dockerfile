FROM public.ecr.aws/lambda/python:3.11

WORKDIR ${LAMBDA_TASK_ROOT}

COPY snorefox_med/api_sqs_bridge/api_sqs_bridge_requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r api_sqs_bridge_requirements.txt

RUN mkdir -p snorefox_med/client
COPY snorefox_med/__init__.py ./snorefox_med
COPY snorefox_med/api_sqs_bridge ./snorefox_med/api_sqs_bridge
COPY snorefox_med/client/__init__.py ./snorefox_med/client
COPY snorefox_med/client/snorefox_med_client ./snorefox_med/client/snorefox_med_client
COPY snorefox_med/client/sqs_client ./snorefox_med/client/sqs_client
COPY snorefox_med/metadata ./snorefox_med/metadata
COPY snorefox_med/utils ./snorefox_med/utils
COPY pyproject.toml .

CMD ["snorefox_med.api_sqs_bridge.entrypoint.lambda_handler"]
