# Contributing

We welcome contributions to the PyTorch KAN project! This document provides guidelines for contributing to the development of this library.

## Setting Up Development Environment

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/your-username/pytorch_kan.git
   cd pytorch_kan
   ```

2. **Set up development environment using Poetry**:
   ```bash
   # Install Poetry if you don't have it
   curl -sSL https://install.python-poetry.org | python3 -

   # Install dependencies including development dependencies
   poetry install --with dev
   ```

3. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

## Development Workflow

1. **Create a new branch for your feature or bugfix**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**: Implement your feature or fix the bug.

3. **Add tests**: Ensure that your code is covered by tests.

4. **Run linting and type checking**:
   ```bash
   # Run all checks with tox
   tox -e lint,type

   # Or run them individually
   black .
   isort .
   flake8
   mypy src
   ```

5. **Run tests**:
   ```bash
   # Run tests with tox to check multiple Python versions
   tox

   # Or run tests directly with pytest
   pytest
   ```

6. **Build documentation locally**:
   ```bash
   # Using tox
   tox -e docs

   # Or directly with sphinx
   cd docs
   make html
   ```

7. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Your descriptive commit message"
   ```

8. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

9. **Create a pull request** from your fork to the main repository.

## Code Style Guidelines

- **Follow PEP 8**: We use Black and flake8 to enforce PEP 8 compliance.
- **Type Hints**: Add type hints to all function definitions.
- **Docstrings**: Write clear docstrings in Google style format.
- **Comments**: Comment complex sections of code, but prefer readable code over excessive comments.

## Documentation Guidelines

- **API Documentation**: Update API documentation when modifying or adding new functions/classes.
- **Examples**: Consider adding examples to demonstrate usage of new features.
- **Tutorials**: For significant features, consider adding a tutorial.

## Testing Guidelines

- **Test Coverage**: Aim for 100% test coverage for new code.
- **Unit Tests**: Write unit tests for all new functions and classes.
- **Integration Tests**: Consider integration tests for features that interact with other components.
- **Property-Based Testing**: Consider using `hypothesis` for property-based testing where appropriate.

## Pull Request Process

1. **PR Description**: Provide a clear description of the changes and the problem they solve.
2. **Linked Issues**: Reference any related issues.
3. **CI Checks**: Ensure all CI checks pass.
4. **Code Review**: Address any feedback from code reviews.
5. **Approval**: Wait for approval from at least one maintainer.

## Adding New Features

When adding new features to PyTorch KAN, please consider the following:

1. **Design**: Discuss major features in an issue first to align with the project's goals.
2. **Dependencies**: Minimize new dependencies.
3. **Backward Compatibility**: Maintain backward compatibility when possible.
4. **Documentation**: Document new features comprehensively.
5. **Examples**: Provide usage examples for new features.

## Adding New Basis Functions

When implementing a new basis function:

1. **Inherit from BaseBasis**: All basis functions should inherit from the `BaseBasis` class.
2. **Implement Required Methods**: At minimum, implement the `calculate_basis` method.
3. **Mathematical Correctness**: Ensure the implementation is mathematically correct.
4. **Numerical Stability**: Consider numerical stability in the implementation.
5. **Documentation**: Document the mathematical properties of the basis function.

## License

By contributing to PyTorch KAN, you agree that your contributions will be licensed under the project's MIT license.