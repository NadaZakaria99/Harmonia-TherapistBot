import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import gradio as gr

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# Set Google API Key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.3,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

CONDITIONED_PROMPT = """
You are Harmonia, a compassionate and inclusive mental health and emotional support guide. Your role is to provide a safe, non-judgmental space for individuals to express their feelings, explore their thoughts, and receive both emotional validation and practical guidance. You are here to support people through a wide range of mental health challenges, from everyday stress to more complex emotional struggles.

**Core Principles for Short Responses:**
1. **Be Concise**: Respond in 2-3 sentences maximum.
2. **Emotional Validation**: Acknowledge the user's feelings briefly.
3. **Practical Guidance**: Offer 1-2 actionable steps or coping strategies.
4. **Empathy**: Maintain a warm and empathetic tone.
5. **Crisis Handling**: If the user is in immediate danger, prioritize clear and direct instructions.

**Response Structure:**
1. **Acknowledge Feelings**: "I hear how [challenging/overwhelming/difficult] this is for you."
2. **Offer Support**: "Here’s something you can try: [1-2 strategies]."
3. **Encourage Next Steps**: "You’re not alone. Let’s take this one step at a time."

**Crisis Protocol:**
- If the user is in immediate danger, respond with:
  - "Your safety is the priority. Please contact [crisis hotline] or [emergency services] right away."
  - Provide grounding techniques if appropriate.

**Examples of Short Responses:**
- "I hear how overwhelming this feels. Try taking deep breaths and focusing on one small step at a time. You’re not alone."
- "It sounds like you’re going through a tough time. Let’s talk about what might help you feel safer. You’ve got this."
- "Your feelings are valid. Here’s a strategy: write down your thoughts to help process them. I’m here to support you."

**Special Considerations:**
- If the user thanks you, respond: "I'm here anytime you need support. Don’t hesitate to reach out."
- If the user says hello, respond: "Hello! How can I support you today? Feel free to share what’s on your mind."

**Remember**: Always adapt the response based on the urgency and specific needs of the situation while keeping it concise and supportive.

Retrieved Information:
{context}
Context from previous conversation:
{chat_history}
Current situation: {question}
"""

# Load PDF
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def create_mental_health_assistant():
    # Load PDF and create splits
    pdf_path = "Harmonia.pdf"  # Update this path to your PDF file
    documents = load_pdf(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)

    # Create embeddings and vectorstore
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="/tmp/chroma_db"
    )

    # Initialize memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    # Modify the prompt template to include context
    PROMPT = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=CONDITIONED_PROMPT
    )

    # Create chain with memory and custom prompt
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=False,
        verbose=True
    )

    return qa_chain

def create_interface(qa_chain):
    def respond(message, history):
        response = qa_chain({"question": message})
        return response["answer"]

    # Custom CSS for the background image and compact examples

    
    
    # Custom CSS for the background image, centering, and scrollable content
  
    # Custom CSS for the background image, centering, and compact layout
   
    # Custom CSS for the background image, centering, and compact chat area
    custom_css = """
    body {
        background-image: url('https://huggingface.co/spaces/Nadaazakaria/Harmonia/resolve/main/Harmonia.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        margin: 0;
        overflow: auto; /* Allow scrolling */
    }
    .gradio-container {
        background-color: rgba(255, 255, 255, 0.8);  /* Semi-transparent white background */
        padding: 10px;  /* Reduced padding */
        border-radius: 10px;
        text-align: center;
        max-width: 600px;  /* Reduced max-width */
        width: 100%;
        max-height: 90vh; /* Limit height to 90% of viewport height */
        overflow-y: auto; /* Enable vertical scrolling */
        margin: 10px; /* Reduced margin */
    }
    .chat-interface {
        padding: 10px;  /* Reduced padding */
    }
    .chat-area {
        max-height: 400px;  /* Reduced height of the chat area */
        overflow-y: auto;   /* Add scroll if content exceeds the height */
        padding: 10px;  /* Reduced padding */
        border: 1px solid #ddd;
        border-radius: 5px;
        background-color: rgba(255, 255, 255, 0.9);  /* Slightly opaque background */
        margin-top: 10px;  /* Adjusted margin */
    }
    .examples {
        max-height: 120px;  /* Further reduced height of the examples section */
        overflow-y: auto;   /* Add scroll if content exceeds the height */
        padding: 5px;  /* Reduced padding */
        border: 1px solid #ddd;
        border-radius: 5px;
        background-color: rgba(255, 255, 255, 0.9);  /* Slightly opaque background */
        margin-top: 10px;  /* Adjusted margin */
    }
    .chatbot-icon {
        width: 40px;  /* Reduced icon size */
        height: 40px;  /* Reduced icon size */
    }
    """
    
    # Create the chat interface with centered title and description
    chat_interface = gr.ChatInterface(
        fn=respond,
        title="",  # Remove the title from here since it's included in the description

        description="""
        <div style="text-align: center;">
            <h1>Harmonia🌸</h1>
            <p>Your peace, your path, your Harmonia</p>
            <p>🌸 Hello, I'm Harmonia. 🌸</p>
            <p>I’m here to provide a safe, non-judgmental space for you to express your feelings, explore your thoughts, and receive support. Whether you're dealing with stress, anxiety, relationship issues, or just need someone to talk to, I'm here to listen and help you navigate your emotions.</p>
            <p>Together, we’ll explore what’s on your mind and work towards finding peace and clarity. Remember, you’re not alone, and it’s okay to ask for help.</p>
            <p>Whenever you’re ready, we can begin. 💬</p>
        </div>
        """,
        examples=[
            "I've been feeling really overwhelmed lately...",
            "I'm struggling with my relationships...",
            "I'm not sure how to deal with my anxiety...",
            "I've been feeling really low and don't know why...",
        ],
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="teal",
        ),
        css=custom_css  # Inject custom CSS
    )
    return chat_interface

# Usage
if __name__ == "__main__":
    # Create assistant with mental health-focused memory
    qa_chain = create_mental_health_assistant()

    # Create and launch interface
    chat_interface = create_interface(qa_chain)
    chat_interface.launch(share=True)
