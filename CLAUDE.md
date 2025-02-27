# CLAUDE.md - Guidelines for AI Podcast Weekly Newsletter Generator

## Commands
- Run script: `python main.py`
- Run with custom environment: `OPENROUTER_API_KEY=your_key python main.py`
- Docker build: `docker build -t dan-mac-ai-podcast-weekly .`
- Docker run: `docker run -e OPENROUTER_API_KEY=your_key dan-mac-ai-podcast-weekly`

## Code Style
- **Imports**: Standard library first, then third-party, then local imports
- **Functions**: Use type hints for parameters and return values
- **Error handling**: Use try/except blocks with specific exception types
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Documentation**: Docstrings for functions with Args and Returns sections
- **Environment**: Use python-dotenv for loading environment variables

## Architecture
- Single script implementation with modular functions
- Core flow: retrieve files → extract information → analyze connections → generate newsletter
- JSON is used as the data exchange format between LLM and application
- Error handling includes logging to console with contextual information