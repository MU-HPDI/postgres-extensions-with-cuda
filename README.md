# CUDA integration with PostgreSQL

This project is a example of a CUDA integration with PostgreSQL. It is a simple example of a CUDA kernel that is executed on the GPU and the result is return as a set returning function (SRF).

## Requirements

- CUDA Toolkit
- PostgreSQL server and development headers

## Installation

1. Clone the repository: `git clone https://github.com/MU-HPDI/postgres-extensions-with-cuda.git`
2. Navigate to the project directory: `cd postgres-extensions-with-cuda`
3. Build the project: `make`

## Usage

1. Make sure that the PostgreSQL server is running.
2. Set the required environment variables in a `.env` file.
3. Create environment for python virtual environment: `python3 -m venv env`
4. Install python dependencies: `pip install -r requirements.txt`
5. Insert dummy data into the database: `make insert`
6. Run the project: `make clean all`
7. Optionally, you can run only the CUDA kernel: `make clean cuda`
   1. Then, you execute the kernel: `./cuda.out`
## Environment Variables

The following environment variables are required to run the project:

- `PGUSER`: The PostgreSQL username
- `PGPASSWORD`: The PostgreSQL password
- `PGHOST`: The PostgreSQL server host name
- `PGPORT`: The PostgreSQL server port number
- `PGDATABASE`: The PostgreSQL database name

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

- [PostgreSQL](https://www.postgresql.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

Please feel free to update and modify this README.md file to suit your specific project needs.
