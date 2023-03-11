FROM ubuntu:latest

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN \
    apt-get update              &&  \
    apt-get upgrade --yes       &&  \
    apt-get install --yes           \
        python3-flask

COPY requirements.txt .
RUN pip install -r requirements.txt

ADD app.py /srv/app.py
ENV FLASK_APP=/srv/app.py
CMD ["flask", "run", "--host=0.0.0.0"]

ADD docker-entrypoint.sh /srv/docker-entrypoint.sh
RUN chmod +x /srv/docker-entrypoint.sh
ENTRYPOINT ["/srv/docker-entrypoint.sh"]