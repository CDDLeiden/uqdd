# Contributing to Our Project

First off, thank you for considering contributing to our project! It's people like you that make our project such a
great tool.

We welcome contributions in various forms, including bug reports, feature requests, documentation improvements, and code
contributions.

## Reporting Bugs

If you encounter a bug, please help us by submitting an issue to our GitHub repository. When you are creating a bug
report, please include as many details as possible. Fill out the required template, the information it asks for helps us
resolve issues faster.

## Suggesting Enhancements

If you have an idea for a new feature or an improvement to an existing one, please submit an issue to our GitHub
repository. Clearly describe the proposed enhancement and its potential benefits.

## Code Contribution Guidelines

We follow a standard Fork & Pull Request workflow.

1. **Fork the repository:** Start by forking the main repository to your GitHub account.
2. **Create a branch:** For any new feature or bug fix, create a new branch in your forked repository. Use a descriptive
   name for your branch (e.g., `fix-login-bug` or `add-new-feature`).
3. **Make your changes:** Make your code changes in your branch. Ensure your code adheres to any existing coding
   standards and include tests if applicable.
4. **Test your changes:** Run the test suite to ensure your changes don't break existing functionality.
5. **Commit your changes:** Write clear and concise commit messages.
6. **Submit a Pull Request (PR):** Push your changes to your forked repository and then open a Pull Request to the main
   repository. Provide a clear description of the changes in your PR.

## Setting up the Development Environment

To set up the development environment, you will need Conda.

1. Clone your forked repository to your local machine.
2. Navigate to the project's root directory.
3. Create and activate the Conda environment using the provided `environments/environment_{OS}.yml` file, for example in Linux:
   ```bash
   conda env create -f environments/environment_linux.yml
   conda activate uqdd-env
   ```

## Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2. Update the README.md with details of changes to the interface, this includes new environment variables, exposed
   ports, useful file locations and container parameters.
3. Increase the version numbers in any examples files and the README.md to the new version that this Pull Request would
   represent. The versioning scheme we use is SemVer.
4. Your PR will be reviewed by a maintainer. They may ask for changes or provide feedback.
5. Once your PR is approved and passes all checks, it will be merged into the main codebase.

Thank you for your contribution!
