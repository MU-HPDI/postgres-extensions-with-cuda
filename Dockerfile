FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
ENV PG_MAJOR_VER=14

RUN apt-get update
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:apt-fast/stable --yes
RUN apt update && apt-get install -y apt-fast
RUN apt-get update && apt-fast install -y \
    libopenblas-dev \
    libssl-dev \
    bison \
    flex \
    pkg-config \
    libreadline-dev \
    libz-dev \
    curl \
    lsb-release \
    tzdata \
    sudo \
    cmake \
    libpq-dev \
    libclang-dev \
    wget \
    postgresql-plpython3-$PG_MAJOR_VER \
    postgresql-$PG_MAJOR_VER \
    postgresql-server-dev-$PG_MAJOR_VER \
    git \ 
    python3 \
    python3-pip


WORKDIR /app

# add postgres user to sudoers
RUN echo 'postgres ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers


WORKDIR /app

RUN chmod a+rwx `$(which pg_config) --pkglibdir` \
    `$(which pg_config) --sharedir`/extension \
    /var/run/postgresql/

RUN pip3 install --upgrade pip
RUN pip3 install torch
RUN pip3 install numpy

RUN useradd pg_extension_user -m -s /bin/bash -G sudo
RUN echo 'pg_extension_user ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers

COPY --chown=postgres:postgres ./ /app
RUN sudo chown -R postgres:postgres /app

RUN sudo cp /app/docker/postgresql.conf /etc/postgresql/$PG_MAJOR_VER/main/postgresql.conf
RUN sudo cp /app/docker/pg_hba.conf /etc/postgresql/$PG_MAJOR_VER/main/pg_hba.conf


RUN sudo chmod +x /app/docker/entrypoint.sh

USER root

RUN mkdir build && \
    make docker

RUN sudo chown -R pg_extension_user:pg_extension_user /usr/share/postgresql/$PG_MAJOR_VER/
RUN sudo chown -R pg_extension_user:pg_extension_user /usr/share/postgresql/$PG_MAJOR_VER/extension


USER postgres

ENV PGDATA /var/lib/postgresql/$PG_MAJOR_VER/main/pgdata

ENTRYPOINT ["/bin/bash", "/app/docker/entrypoint.sh"]

# We set the default STOPSIGNAL to SIGINT, which corresponds to what PostgreSQL
# calls "Fast Shutdown mode" wherein new connections are disallowed and any
# in-progress transactions are aborted, allowing PostgreSQL to stop cleanly and
# flush tables to disk, which is the best compromise available to avoid data
# corruption.
#
# Users who know their applications do not keep open long-lived idle connections
# may way to use a value of SIGTERM instead, which corresponds to "Smart
# Shutdown mode" in which any existing sessions are allowed to finish and the
# server stops when all sessions are terminated.
#
# See https://www.postgresql.org/docs/12/server-shutdown.html for more details
# about available PostgreSQL server shutdown signals.
#
# See also https://www.postgresql.org/docs/12/server-start.html for further
# justification of this as the default value, namely that the example (and
# shipped) systemd service files use the "Fast Shutdown mode" for service
# termination.
#
STOPSIGNAL SIGINT
#
# An additional setting that is recommended for all users regardless of this
# value is the runtime "--stop-timeout" (or your orchestrator/runtime's
# equivalent) for controlling how long to wait between sending the defined
# STOPSIGNAL and sending SIGKILL (which is likely to cause data corruption).
#
# The default in most runtimes (such as Docker) is 10 seconds, and the
# documentation at https://www.postgresql.org/docs/14/server-start.html notes
# that even 90 seconds may not be long enough in many instances.
CMD ["/usr/lib/postgresql/14/bin/postgres", "-c", "config_file=/etc/postgresql/14/main/postgresql.conf"]