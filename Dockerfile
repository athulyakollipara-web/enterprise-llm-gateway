# One image for local Docker and AWS Lambda. On Lambda, the AWS Lambda Web Adapter (an extension)
# turns API Gateway events into HTTP requests to uvicorn; outside Lambda it does nothing.
FROM public.ecr.aws/docker/library/python:3.12-slim
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.0.0 /lambda-adapter /opt/extensions/lambda-adapter

ENV PORT=8080 \
    AWS_LWA_READINESS_CHECK_PATH=/healthz \
    GATEWAY_HOME=/tmp/llmgw \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY llmgw ./llmgw
RUN pip install --no-cache-dir . && useradd --create-home gateway
USER gateway

EXPOSE 8080
CMD ["uvicorn", "llmgw.app:app", "--host", "0.0.0.0", "--port", "8080"]
