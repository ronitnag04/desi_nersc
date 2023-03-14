FROM ubuntu:latest

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN \
    apt-get update              &&  \
    apt-get upgrade --yes       &&  \
    apt-get install --yes           \
        python3-flask               \
        python3-astropy             \
        python3-matplotlib          \
        python3-numpy               \
        python3-sqlalchemy     

ADD app.py /srv/app.py
ENV FLASK_APP=/srv/app.py
CMD ["flask", "run", "--host=0.0.0.0"]

ADD docker-entrypoint.sh /srv/docker-entrypoint.sh
RUN chmod +x /srv/docker-entrypoint.sh
ENTRYPOINT ["/srv/docker-entrypoint.sh"]