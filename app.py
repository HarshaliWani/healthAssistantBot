from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import streamlit as st

MODEL_NAME = "google/flan-t5-base"  # Add model name

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

def generate_response(model, tokenizer, user_input):
    
    prompt = (
        f"You are a knowledgeable and detailed healthcare assistant. "
        f"Please provide a comprehensive and helpful answer to the following question: {user_input}"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)

    # Generate response
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        min_new_tokens=20,  # Ensure a minimum length
        max_new_tokens=250,  # Allow for longer responses
        temperature=0.5,  # Lower randomness for focused answers
        top_k=50,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.7,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    st.title("AI Healthcare Assistant")
    st.write("Ask me anything about your health!")

    tokenizer, model = load_model()

    # Initialize session state for conversation history
    if "conversation" not in st.session_state:
        st.session_state.conversation = []  

    user_input = st.text_input("Your question:")
    submit_button = st.button("Submit")

    if submit_button and user_input.strip():
        with st.spinner("Generating response..."):
            response = generate_response(model, tokenizer, user_input)
        # Add the new question and response to the conversation history
        st.session_state.conversation.append((user_input, response))
    
    st.write("### Conversation:")
    for idx, (question, answer) in enumerate(reversed(st.session_state.conversation)):
        st.write(f"**Q{len(st.session_state.conversation) - idx}:** {question}")
        st.write(f"**A{len(st.session_state.conversation) - idx}:** {answer}")
        st.write("---")  

if __name__ == "__main__":
    main()
