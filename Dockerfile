FROM public.ecr.aws/lambda/python:3.12

# Copied separately from the rest of the code so Docker can cache this layer -
# rebuilding after a code-only change won't re-run pip install.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY scripts ./scripts
COPY lambda_handler.py .

# AWS calls handler(event, context) - the Lambda Runtime Interface Client
# baked into this base image is what makes that contract work.
CMD ["lambda_handler.handler"]