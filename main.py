import streamlit as st
from langchain_helper import query_ans, get_few_shot_db_chain



def initialize_session_state():  # Added persistence for Chroma vectorstore
    if 'chain' not in st.session_state:
        st.session_state.chain = get_few_shot_db_chain()
        
        

def main():
    st.title("AtliQ T Shirts: Database Q&A 👕")
    
    # Initialize session state
    initialize_session_state()
    
    question = st.text_input("Question: ")

    if question:
        try:
            response = st.session_state.chain.invoke({"question": question})
            
            st.header("Answer")
            st.write(response)
            
            answer = query_ans(response)
            if answer:
                st.write(f"Extracted Answer: {answer}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.button("Reset Chain", on_click=lambda: st.session_state.pop('chain', None))
            
            
            

if __name__ == "__main__":
    main()
    
    
# mkdir chroma_db

# streamlit run main.py    