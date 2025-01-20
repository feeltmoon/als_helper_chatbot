import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import sqlite3
from openai import OpenAI
import json
import os


# Retrieve the API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
    st.stop()
# Set the API key for OpenAI
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Initialize OpenAI client
client = OpenAI()

# ... [rest of your existing code remains unchanged] ...
# Function to connect to the SQLite database
def connect_db():
    return sqlite3.connect("local_data.db")

# Format the retrieved data dynamically with column names, adding line-breaks for each column-value pair
def format_results_conditional(results, table_name, conn):
    """
    Format query results conditionally: Add column names only if the results represent full rows.
    """
    # Retrieve the table schema (column names)
    table_schema = get_column_names_outside(conn, table_name)
    table_column_count = len(table_schema)

    formatted_result = ""
    for index, row in enumerate(results):
        # Check if the result row matches the full column count
        if len(row) == table_column_count:
            # Map column names to row values
            row_data = zip(table_schema, row)
            row_text = "\n\n".join([f"**{col}**: {value}" for col, value in row_data])
        else:
            # Display row values without column names
            row_text = "\n\n".join([str(value) for value in row])
        
        formatted_result += f"**Row {index}:**\n{row_text}\n\n{'*' * 40}\n\n"
    return formatted_result.strip()

# Function to get column names for a table
def get_column_names_outside(conn, table_name):
    """Return a list of column names for a given table."""
    return [col[1] for col in conn.execute(f"PRAGMA table_info('{table_name}');").fetchall()]

# Function to retrieve database schema
def get_database_schema(conn):
    def get_table_names(conn):
        """Return a list of table names."""
        return [table[0] for table in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]

    def get_column_names(conn, table_name):
        """Return a list of column names."""
        return [col[1] for col in conn.execute(f"PRAGMA table_info('{table_name}');").fetchall()]

    schema = []
    for table in get_table_names(conn):
        schema.append({"table_name": table, "columns": get_column_names(conn, table)})
    return schema

# Function to query the SQLite database
def ask_database(conn, query):
    """Function to query SQLite database with a provided SQL query."""
    try:
        result = conn.execute(query).fetchall()
        return result
    except Exception as e:
        return f"Error: {e}"

def chatbot_interaction(messages, database_schema_string):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "ask_database",
                "description": "Query the SQLite database using SQL commands.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"""
                                SQL query extracting info to answer the user's question.
                                If the user specifies multiple values for a table, use `WHERE column_name IN ('value1', 'value2')`.
                                For example:
                                - For multiple values in one table: SELECT * FROM table_name WHERE column_name IN ('value1', 'value2');
                                SQL should be written using this database schema:
                                {database_schema_string}
                                The query should be returned in plain text, not in JSON.
                                """,
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    # Debug: Check messages structure before making the API call
    #import json
    print(json.dumps(messages, indent=2))
    # Interact with OpenAI ChatGPT with tools
    
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    return response

# Intro Section
def intro():
    st.write("# Welcome to Your Streamlit App! 👋")
    st.markdown(
        """
        This application allows you to **Upload ALS Excel files** and **Query the ChatBot**.

        ### How to Use:
        - **Upload ALS**: Navigate to the "Upload ALS" section to upload your ALS Excel files and save them to the SQLite database.
        - **Query ChatBot**: Navigate to the "Query ChatBot" section to interact with the ChatBot integrated with your database.

        Use the sidebar to select the desired option.
        """
    )

# Main App Section for Uploading ALS
def upload_als():
    st.title("Upload ALS Excel and Save to SQLite")

    # File uploader
    uploaded_file = st.file_uploader("Upload XLSX file named ALS", type="xlsx")
    if uploaded_file is not None:
        # Read all sheets into a dictionary of DataFrames
        all_sheets = pd.read_excel(uploaded_file, sheet_name=None, engine='openpyxl')

        # Define which sheets we actually care about
        required_sheets = [
            "Forms",
            "Fields",
            "Folders",
            "DataDictionaries",
            "DataDictionaryEntries",
            "Checks",
            "CheckSteps",
            "CheckActions",
            "Derivations",
            "DerivationSteps",
            "CustomFunctions"
        ]

        # Create or connect to an SQLite database
        engine = create_engine("sqlite:///local_data.db")

        st.write("All sheets found in the uploaded file:")
        st.write(list(all_sheets.keys()))

        # Dictionary to store successfully processed DataFrames
        processed_dfs = {}

        # Iterate through the list of required_sheets, process only if present
        for sheet_name in required_sheets:
            if sheet_name in all_sheets:
                df = all_sheets[sheet_name]
                # Save the DataFrame to SQLite
                df.to_sql(sheet_name, con=engine, if_exists="replace", index=False)
                st.success(f"Sheet '{sheet_name}' has been written to the SQLite database!")
                processed_dfs[sheet_name] = df  # Save DataFrame for later display
            else:
                st.warning(f"'{sheet_name}' not found in the uploaded file; skipping.")

        st.write("Done processing all required sheets.")

        # Display radio buttons for each sheet and show corresponding DataFrame
        if processed_dfs:
            selected_sheet = st.radio(
                "Select a sheet to view its content:",
                horizontal=True,
                options=list(processed_dfs.keys())
            )
            if selected_sheet:
                st.write(f"Displaying first 5 rows of the '{selected_sheet}' sheet:")
                st.dataframe(processed_dfs[selected_sheet].head())

# Main ChatBot Section
def query_chatbot():
    st.title("Chatbot with Database Integration")
    conn = connect_db()
    database_schema = get_database_schema(conn)
    database_schema_string = "\n".join(
        [f"Table: {table['table_name']} Columns: {', '.join(table['columns'])}" for table in database_schema]
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful assistant with database capabilities."}
        ]

    # Display messages in the chat interface
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg.get("content", ""))

    if user_input := st.chat_input("Ask me anything"):
        # Append the user input to session messages
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Prepare assistant response
        with st.chat_message("assistant"):
            try:
                # Construct minimal valid context for GPT
                context_messages = [
                    {"role": "system", "content": "You are a helpful assistant with database capabilities. When the user asks for multiple values, use SQL `IN` clauses for querying."},
                    {"role": "user", "content": user_input}
                ]

                # Debugging: Print context being sent to the API
                import json
                print("Messages sent to API:")
                print(json.dumps(context_messages, indent=2))

                # Interact with ChatGPT
                response = chatbot_interaction(context_messages, database_schema_string)

                # Extract assistant's response
                ai_message = response.choices[0].message.content
                st.markdown(ai_message)
                st.session_state.messages.append({"role": "assistant", "content": ai_message})

                # Handle tool invocations if present
                tool_calls = response.choices[0].message.tool_calls
                if tool_calls:
                    tool_call_id = tool_calls[0].id
                    tool_function_name = tool_calls[0].function.name
                    tool_query_string = json.loads(tool_calls[0].function.arguments)["query"]
                    
                    print(f"Generated SQL Query: {tool_query_string}")  # Log query for debugging
                    st.markdown(f"### Generated SQL Query:\n```sql\n{tool_query_string}\n```")

                    if tool_function_name == "ask_database":
                        # Query the database
                        results = ask_database(conn, tool_query_string)
                        
                        query_table = tool_query_string.split("FROM")[1].split()[0].strip()  # Extract table name
                        # Format results conditionally based on column count
                        formatted_results = format_results_conditional(results, query_table, conn)

                        # Append and display formatted results
                        st.session_state.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_function_name,
                            "content": formatted_results
                        })
                        #st.markdown(f"**Query Results:**\n\n{formatted_results}")
                        # Use Markdown to display formatted results with line-breaks
                        st.markdown(f"### Query Results:\n\n{formatted_results}", unsafe_allow_html=False)
            except Exception as e:
                st.markdown(f"Error: {e}")

                # Debugging the error
                print(f"Error: {e}")

# Intro Section Example (optional enhancements can be added here)
def intro_section():
    st.title("Welcome to Your Streamlit App")
    st.markdown(
        """
        ### Available Features:
        - **Upload ALS**: Upload and process ALS Excel files.
        - **Query ChatBot**: Interact with the ChatBot integrated with your database.

        Use the sidebar to navigate between these features.
        """
    )

# Mapping of page names to functions
page_names_to_funcs = {
    "—": intro,             # Intro section displayed when no option is selected
    "Upload ALS": upload_als,
    "Query ChatBot": query_chatbot
}

# Sidebar Navigation using selectbox
selected_page = st.sidebar.selectbox("Choose an option", options=list(page_names_to_funcs.keys()))

# Display the selected page
page_names_to_funcs[selected_page]()
