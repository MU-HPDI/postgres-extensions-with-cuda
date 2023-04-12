FROM nvidia/cuda:12.1.0-devel-ubuntu18.04

WORKDIR /app
COPY . /app 
COPY .env /app/.env


# Set the DEBIAN_FRONTEND environment variable
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install dependencies
RUN apt-get update && \
    apt-get install -y wget gnupg lsb-release && \
    wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - && \
    echo "deb http://apt.postgresql.org/pub/repos/apt/ $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list && \
    apt-get update && \
    apt-get install -y postgresql-14 postgresql-contrib && \
    apt-get install -y postgresql-server-dev-14

# run make clean all from the Makefile
# Switch back to the root user
USER root
RUN make insert
RUN make docker

# Create a new Postgres user and database
USER postgres
RUN /etc/init.d/postgresql start && \
    psql -f /app/script.sql && \
    psql --command "CREATE USER docker WITH SUPERUSER PASSWORD 'docker';" && \
    createdb -O docker docker 

# Allow remote connections to Postgres
RUN echo "host all  all    0.0.0.0/0  md5" >> /etc/postgresql/14/main/pg_hba.conf && \
    echo "listen_addresses='*'" >> /etc/postgresql/14/main/postgresql.conf

# Expose Postgres port
EXPOSE 5432

# Start Postgres
CMD ["/usr/lib/postgresql/14/bin/postgres", "-D", "/var/lib/postgresql/14/main", "-c", "config_file=/etc/postgresql/14/main/postgresql.conf"]