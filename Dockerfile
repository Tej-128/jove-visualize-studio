FROM python:3.11-slim
WORKDIR /app
COPY studio/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY studio /app/studio
RUN useradd --create-home --uid 10001 studio && mkdir /data && chown studio:studio /data
USER studio
ENV STUDIO_JOBS_ROOT=/data/jobs
EXPOSE 8501
CMD ["streamlit", "run", "studio/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
