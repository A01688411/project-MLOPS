# Pre-Commits
Pre-Commits are a software development practice that consists of performing a series of automatic checks before committing changes to a version control repository.

## Steps to activate Pre-Commits
1. Make sure you have pre-commit installed in your development environment. You can install it using the following command on the command line:
    ```bash
    pip install pre-commit
    ```
2. In the root directory of your project, create a file called ´´´.pre-commit-config.yaml´´´.
3. Open the .pre-commit-config.yaml file and add the following content:
    ```bash
    repos:
        - repo: https://github.com/psf/black
        rev: stable
        hooks:
            - id: black

        - repo: https://github.com/PyCQA/isort
        rev: 5.9.3
        hooks:
            - id: isort

        - repo: https://github.com/PyCQA/flake8
        rev: 3.9.2
        hooks:
            - id: flake8

    exclude: '\.venv|\.git|\.tox|\.mypy_cache|\.pytest_cache|\.eggs|build|dist'
    ```


