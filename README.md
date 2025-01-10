# LangChain Local Run with Llama

Example of conversational AI in **LangChain**, integrated with a locally hosted **Llama-2 model**, and capable of querying document-based knowledge bases.
FastAPI powers the backend, so this can be called from a chatbot app.

Adapt to your purpose, e.g., querying internal sensitive docs.

## Features
- **Local LLM**: Leverages a locally hosted Llama-2 model for privacy and cost-efficient inference.
- **Document Querying**: Loads, processes, and indexes documents in multiple formats:
  - PDF
  - Word
  - CSV
  - PowerPoint
- **Tools and Domains**:
- Have multiple tools to improve doc retrieval
- **Session-Based Memory**: Retains conversation history for context-aware interactions.
- **Custom Agent Prompt**: Provides step-by-step reasoning and decision-making.


## Setup

### **Required Directories**
1. **`chat_model/` Directory**:
   > Contains the Llama-2 model file (e.g., `openhermes-2.5-mistral-7b.Q5_K_M.gguf`).
   > Ensure the model is compatible with `ChatLlamaCpp`.

2. **`docs_to_query/` Directory**:
   > Subdirectories to organize your documents by domain:
   e.g.
   > - `support/` for IT-related documents.
   > Place the documents you want to query here in supported formats (PDF, Word, CSV, PowerPoint).
