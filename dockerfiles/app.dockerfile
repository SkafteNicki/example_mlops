FROM base:latest

EXPOSE $PORT
CMD exec uvicorn src.example_mlops.app:app --host 0.0.0.0 --port $PORT
