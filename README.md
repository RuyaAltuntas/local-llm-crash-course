# ValorCare AI - Veteran Support Assistant

A Chainlit-based AI chatbot designed to help family members, partners, and friends of military veterans understand and support their loved ones. Powered by the Groq LLM API for fast, intelligent responses.

## Overview

**ValorCare AI** provides compassionate, evidence-based guidance for people supporting veterans. The assistant can discuss:
- Veteran mental and physical health topics
- Daily life and well-being
- Communication strategies
- Emotional support approaches
- General veteran-related questions

### Important Note
This tool provides **general informational support only**. It does not have access to personal or medical information about specific individuals, and cannot provide medical diagnoses or treatment recommendations. Users should always consult qualified healthcare professionals for specific medical or mental health concerns.

## Features

✨ **File Upload Support**: Extract and analyze text from PDF, DOCX, and XLSX files  
⚡ **Fast AI Responses**: Powered by Groq's high-performance LLM API  
💬 **Conversational Interface**: Built with Chainlit for an intuitive chat experience  
🛡️ **Veteran-Focused**: Specialized prompts for supporting veterans and their families  
🔐 **Secure**: Uses environment variables for API key management  

## Tech Stack

- **Python 3.8+**
- **Chainlit 1.0.200** - Modern chat UI framework
- **Groq API** - Fast LLM inference
- **FastAPI** - Backend framework
- **PyPDF2, python-docx, openpyxl** - File processing libraries
- **python-dotenv** - Environment variable management

## Installation

### Prerequisites
- Python 3.8 or higher
- A Groq API key (get one at [console.groq.com](https://console.groq.com))

### Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd local-llm-crash-course
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root directory:
   ```env
   GROQ_API_KEY=gsk_your_api_key_here
   ```

## Usage

### Running the Application

Start the Chainlit app:
```bash
chainlit run chatex.py
```

The application will start at `http://localhost:8000` by default. Open it in your browser to begin chatting.

### How to Use

1. **Start a conversation** - Ask any question about supporting a veteran
2. **Upload files** - Share PDF, DOCX, or XLSX files for analysis
3. **Get guidance** - Receive clear, empathetic responses tailored to your needs

### Example Questions

- "How can I better understand my partner's PTSD symptoms?"
- "What are some communication strategies with a veteran?"
- "How do I support a veteran dealing with mental health challenges?"

## Project Structure

```
local-llm-crash-course/
├── chatex.py           # Main Chainlit application
├── chainlit.md         # Welcome message for the chat UI
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this file)
└── README.md          # This file
```

### File Descriptions

- **chatex.py** - Core application containing:
  - Groq API client initialization
  - File extraction functions (PDF, DOCX, XLSX)
  - Message building logic with system prompts
  - Chainlit event handlers for chat

- **chainlit.md** - Custom welcome message displayed when the app starts

- **requirements.txt** - List of all Python package dependencies

## Configuration

### Groq API Key
The application requires a valid Groq API key. Set it in your `.env` file:
```env
GROQ_API_KEY=gsk_...
```

If the key is not found, the application will raise an error with setup instructions.

### Customizing the Welcome Message
Edit `chainlit.md` to change the welcome message displayed to users on startup.

### Modifying the System Prompt
The AI assistant's behavior is controlled by the system prompt in `build_messages()` function in `chatex.py`. Adjust the guidelines and instructions there to customize responses.

## Development

### Adding New Features

- **New file types**: Extend `extract_text_from_file()` function in `chatex.py`
- **Custom handlers**: Use Chainlit decorators like `@cl.on_message`, `@cl.on_chat_start`
- **System behavior**: Modify the system prompt in `build_messages()`

### Testing

To test file extraction:
```python
from chatex import extract_text_from_file
text = extract_text_from_file("path/to/file.pdf")
print(text)
```

## Troubleshooting

### Issue: "GROQ_API_KEY is not set"
- **Solution**: Create a `.env` file with `GROQ_API_KEY=gsk_...`

### Issue: Application won't start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

### Issue: File upload fails
- Only PDF, DOCX, and XLSX files are supported
- Ensure the file is not corrupted
- Check file permissions

## Important Disclaimers

⚠️ **Not a Medical Service**: This tool provides general information and support strategies, not medical advice.

⚠️ **Crisis Support**: If someone mentions self-harm, suicide, or immediate danger, encourage them to contact emergency services or a crisis hotline immediately.

⚠️ **No Personal Data Access**: The AI does not access or store personal medical records or sensitive information about specific individuals.

## Resources for Veterans and Families

- [Veterans Crisis Line](https://www.veteranscrisisline.net/) - 988 then press 1
- [VA Mental Health Services](https://www.mentalhealth.va.gov/)
- [National Alliance on Mental Illness (NAMI)](https://www.nami.org/)
- [Wounded Warrior Project](https://www.woundedwarriorproject.org/)

## License

[Add your license information here]

## Support

For issues, questions, or contributions, please [provide contact information or GitHub link].

---

**Made with ❤️ to support those who support our veterans.**


