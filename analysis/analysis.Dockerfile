FROM public.ecr.aws/lambda/python:3.11

WORKDIR ${LAMBDA_TASK_ROOT}

RUN yum update -y && \
    yum install -y libsndfile sox && \
    yum clean all && \
    rm -rf /var/cache/yum

COPY snorefox_med/analysis/analysis_requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r analysis_requirements.txt

RUN mkdir -p snorefox_med

COPY snorefox_med/__init__.py ./snorefox_med
COPY snorefox_med/analysis ./snorefox_med/analysis
COPY snorefox_med/shared_analysis_report ./snorefox_med/shared_analysis_report
COPY snorefox_med/client/s3_client ./snorefox_med/client/s3_client
COPY snorefox_med/client/snorefox_med_client ./snorefox_med/client/snorefox_med_client
COPY snorefox_med/patient ./snorefox_med/patient
COPY snorefox_med/scoring ./snorefox_med/scoring
COPY snorefox_med/utils ./snorefox_med/utils
COPY pyproject.toml .

CMD ["snorefox_med.analysis.entrypoint.lambda_handler"]
