FROM python:3.8.12-slim-buster

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
COPY  data /data
COPY model_weights /model_weights
COPY  src /src


ENV PYTHONPATH=.
EXPOSE 8000

WORKDIR .

CMD ["python3", "src/predict_team_rest_server.py", "-l", "model_weights/team_clf.pth", "-t", "data/preprocessed/team_membership.json"]

