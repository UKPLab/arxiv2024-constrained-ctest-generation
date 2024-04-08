# User Study Interface

This implements the interface used in our user study. For the reviewing, all links and names that may lead to deanonymization have been removed. The user study runs on a [Flask](https://flask.palletsprojects.com/en/3.0.x/) application with a database connected via [SQLAlchemy](https://www.sqlalchemy.org/).

To run the study, first create a virtual environment (e.g., conda) and install the required packages:

    conda create --name=study python=3.10
    conda activate study
    pip install -r requirements.txt

Next, create a database (with an example user `admin` with the password `admin`):

    mysql -u admin -p
    CREATE DATABASE c-test;

Then import the database structure (including c-tests and selection)

    mysql -u admin -p c-test < c-test.sql

The application can be started via:

    cd c_test
    python __init__.py


Exporting data from the database can be done via:

    mysqldump -u admin -p c-test --add-drop-table > c-test.sql

