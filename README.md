# CUDA integration with PostgreSQL

This project is a example of a CUDA integration with PostgreSQL. It is a simple example of a CUDA kernel that is executed on the GPU and the result is return as a set returning function (SRF).

## Requirements

- CUDA Toolkit
- PostgreSQL server and development headers

## Installation

1. Clone the repository: `git clone https://vcs.missouri.edu/jas8dz/spi-integration-cuda`
2. Navigate to the project directory: `cd spi-integration-cuda`
3. Build the project: `make`

## Usage

1. Make sure that the PostgreSQL server is running.
2. Set the required environment variables in a `.env` file.
3. Run the project: `make all`

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
