#!/bin/bash

echo "PGUSER: $PGUSER"

alias psql="sudo -u postgres psql"

function _create_users_n_databases(){
	psql -c "ALTER USER ${PGUSER} with encrypted password '${PGPASSWORD}'" template1
	psql -f scripts/script.sql -P pager $PGDATABASE
}

function _main(){

	sudo chmod 750 $PGDATA

	if [[ -z "$(ls -A $PGDATA)" ]]; then
		echo "Init db"
		/usr/lib/postgresql/14/bin/initdb -D $PGDATA
	fi

	service postgresql start
	_create_users_n_databases
	service postgresql stop

	# test if CUDA is working
	python3 -c '
import torch
print(f"CUDA is available: {torch.cuda.is_available()}")'

}


_main


echo "Ready!"
if [[ ! -z $@ ]]; then
	echo
	echo "To connect to the database: "
	echo "  psql postgres://user:password@host:port/dbname"
	echo
	$@
fi

