# Contributing to GaitSim Assist

Thank you for considering contributing to GaitSim Assist! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We welcome contributions from everyone, regardless of background or experience level.

## How to Contribute

### Reporting Bugs

If you find a bug in the code, please report it by opening an issue on GitHub. When reporting a bug, please include:

- A clear and descriptive title
- A detailed description of the bug
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Any relevant logs or error messages
- Your operating system and Python version

### Suggesting Features

If you have an idea for a new feature, please open an issue on GitHub. When suggesting a feature, please include:

- A clear and descriptive title
- A detailed description of the feature
- Why you think the feature would be useful
- Any relevant examples or mockups

### Pull Requests

If you want to contribute code to the project, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes
4. Run the tests to ensure your changes don't break anything
5. Submit a pull request

When submitting a pull request, please include:

- A clear and descriptive title
- A detailed description of the changes
- Any relevant issue numbers

## Development Setup

To set up a development environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gaitsim-assist.git
   cd gaitsim-assist
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate.bat
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. Run the tests:
   ```bash
   pytest
   ```

## Coding Standards

Please follow these coding standards when contributing to the project:

- Use [PEP 8](https://www.python.org/dev/peps/pep-0008/) for code style
- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for documentation
- Write tests for new features
- Keep functions and methods small and focused
- Use meaningful variable and function names

## License

By contributing to this project, you agree that your contributions will be licensed under the project's [MIT License](LICENSE). 