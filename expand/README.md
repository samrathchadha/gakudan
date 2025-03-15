# Expand Project

A hierarchical prompt exploration system using Google's Generative AI.

## Overview

This project creates a graph-based thought exploration system that can help solve complex problems through hierarchical prompt exploration. It uses Google's Gemini models to:

1. Break down a main problem into multiple perspectives
2. Explore each perspective with specialized agent roles
3. Recursively explore deeper insights
4. Utilize a RAG (Retrieval Augmented Generation) system to reference past responses
5. Combine all insights into a comprehensive solution

## Project Structure

```
expand_project/
├── __init__.py                # Package initialization
├── main.py                    # Entry point with main function
├── knowledge_base.py          # Vector database for semantic search
├── prompt_graph.py            # Graph data structure with visualization
├── rate_limiter.py            # Rate limiting implementation
├── system_prompts.py          # System prompts configuration
└── prompt_processor.py        # Main processor class with worker functions
```

## Features

- **Rate Limiting**: Controls API call frequency to stay within quota limits
- **Knowledge Base**: Uses TF-IDF and cosine similarity for semantic search of past responses
- **Graph-Based Tracking**: Maintains relationships between prompts and responses
- **Hierarchical Exploration**: Recursive exploration with depth control
- **RAG Integration**: Retrieves and incorporates relevant past context
- **Graph Visualization**: Save graph structure for visualization

## Usage

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Set your Gemini API key:
   ```
   export GEMINI_API_KEY="your-api-key"
   ```

3. Run the main script:
   ```
   python -m expand_project.main
   ```

4. Enter a complex problem when prompted, and the system will explore multiple approaches to solving it.

5. The system creates a JSON file (`expand.json`) that can be visualized using the `visualizer.html` file.

## Visualization

The `expand.json` file created during processing can be loaded into `visualizer.html` to see the graph structure of the thought process, including hierarchy and RAG connections.

## Requirements

- Python 3.8+
- Google Generative AI package
- scikit-learn
- numpy
- networkx
- matplotlib
- colorlog
- rich

Install all dependencies using:
```
pip install -r requirements.txt
```

## Notes on Running

When you first run the application, it will:

1. Ask for a problem that requires planning
2. Generate multiple perspectives to analyze the problem
3. Explore each perspective recursively to a configured depth
4. Incorporate relevant insights from previous explorations (RAG)
5. Combine all insights into a comprehensive synthesis
6. Save the thought graph to `expand.json` for visualization

## Customization

You can customize various aspects of the system:

- Modify system prompts in `system_prompts.py`
- Adjust rate limiting parameters in `rate_limiter.py`
- Change exploration depth in `prompt_processor.py` (set `self.max_depth`)
- Tune vector search parameters in `knowledge_base.py`

## Visualization

The included `visualizer.html` (placed in the same directory as `expand.json`) provides an interactive graph visualization of the thought process:

1. Open `visualizer.html` in a web browser
2. Click "Choose File" and select the generated `expand.json` file
3. Explore the graph - hierarchy connections (solid) and RAG connections (dashed)
4. Hover over nodes to see prompt details
5. Click nodes to see their complete content and connections

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Google Gemini API for generative capabilities
- NetworkX for graph data structures
- Rich for improved console output
- scikit-learn for vector similarity