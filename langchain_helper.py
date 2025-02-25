import os
from langchain.chains import create_sql_query_chain
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, text
import google.generativeai as genai
import pymysql
import logging
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX
from few_shots import few_shots
from dotenv import load_dotenv



# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")
api_key = os.getenv("GOOGLE_API_KEY")

# Create database engine and connection
engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
db = SQLDatabase(engine, sample_rows_in_table_info=3)

# Initialize global variables
vectorstore = None
chain = None




def initialize_vectorstore():
    """Initialize the vector store with examples"""
    global vectorstore
    
    if vectorstore is None:
        try:
            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            to_vectorize = [" ".join(example.values()) for example in few_shots]
            vectorstore = Chroma.from_texts(
                to_vectorize, 
                embeddings, 
                metadatas=few_shots,
                persist_directory="./chroma_db"  # Add persistence
            )
            logger.info("Vectorstore initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vectorstore: {e}")
            
        
        
        

def get_few_shot_db_chain():
    """Create and return the database chain"""
    global chain
    
    if chain is None:
        try:
            # Initialize LLM
            api_key = 'AIzaSyBiZ6Hx2av8lRNQCSSZXbUpykwi_Lpmbvc'
            llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

            # Initialize vectorstore if not already done
            initialize_vectorstore()

            # Create example selector
            example_selector = SemanticSimilarityExampleSelector(
                vectorstore=vectorstore,
                k=2,
            )

            # Define the prompt template
            mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
            Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
            Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
            Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
            Pay attention to use CURDATE() function to get the current date, if the question involves "today".
            
            Use the following format:
            
            Question: Question here
            SQLQuery: Query to run with no pre-amble
            SQLResult: Result of the SQLQuery
            Answer: Final answer here
            
            No pre-amble.
            """

            example_prompt = PromptTemplate(
                input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
                template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
            )

            few_shot_prompt = FewShotPromptTemplate(
                example_selector=example_selector,
                example_prompt=example_prompt,
                prefix=mysql_prompt,
                suffix=PROMPT_SUFFIX,
                input_variables=["input", "table_info", "top_k"],
            )

            chain = create_sql_query_chain(llm, db, prompt=few_shot_prompt)
            logger.info("Chain created successfully")
            
        except Exception as e:
            logger.error(f"Error creating chain: {e}")
            raise

    return chain




def query_ans(response):
    """Execute SQL query and return results"""
    try:
        clean_query = response.strip("```sql").strip("```").strip()
        logger.info(f"Executing query: {clean_query}")

        with engine.connect() as conn:
            result = conn.execute(text(clean_query)).fetchone()
            
            if result and result[0] is not None:
                answer = str(result[0])
                logger.info(f"Query result: {answer}")
                return answer
            else:
                logger.warning("No result found")
                return "No result found"
                
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return f"Error executing query: {str(e)}"