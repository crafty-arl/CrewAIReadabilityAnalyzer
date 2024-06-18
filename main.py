import os
import sqlite3
import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool
from pydantic import BaseModel, Field

# Set up API keys if using OpenAI
os.environ["OPENAI_API_KEY"] = "sk-proj-XMJQK0zKC9KNq6Kksu0FT3BlbkFJNnhhzlpennplDfhEoF5A"

# Custom Calculator Tool
class CalculatorTool(BaseTool):
    name: str = "Calculator Tool"
    description: str = "Tool to calculate readability percentage score."

    def _run(self, text: str) -> str:
        print(f"Running CalculatorTool with text: {text}")  # Debug statement
        word_count = len(text.split())
        readability_score = (100 - (word_count / 2))  # Example calculation
        print(f"Calculated Readability Score: {readability_score}%")  # Debug statement
        return f"Readability Score: {readability_score}%"

    async def _arun(self, text: str) -> str:
        return self._run(text)

# Custom SQLite Database Tool
class SQLiteDatabaseTool(BaseTool, BaseModel):
    connection: sqlite3.Connection = Field(default=None, exclude=True)
    cursor: sqlite3.Cursor = Field(default=None, exclude=True)
    db_name: str

    def __init__(self, db_name: str = 'readability.db', **kwargs):
        super().__init__(**kwargs)
        self.db_name = db_name
        self.connection = sqlite3.connect(self.db_name)
        self.cursor = self.connection.cursor()
        self.create_tables()

    def create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY,
                key TEXT UNIQUE,
                result TEXT
            )
        ''')
        self.connection.commit()

    def store_result(self, key: str, result: str):
        self.cursor.execute('''
            INSERT OR REPLACE INTO reports (key, result) VALUES (?, ?)
        ''', (key, result))
        self.connection.commit()

    def retrieve_result(self, key: str) -> str:
        self.cursor.execute('''
            SELECT result FROM reports WHERE key = ?
        ''', (key,))
        row = self.cursor.fetchone()
        return row[0] if row else ''

    def _run(self, action: str, key: str = '', result: str = '') -> str:
        if action == 'store':
            self.store_result(key, result)
            return f"Stored result for key: {key}"
        elif action == 'retrieve':
            return self.retrieve_result(key)
        else:
            return "Invalid action"

    async def _arun(self, action: str, key: str = '', result: str = '') -> str:
        return self._run(action, key, result)

# Initialize the SQLite Database Tool
db_tool = SQLiteDatabaseTool()

# Readability Analyzer Agent
readability_analyzer = Agent(
    role='Readability Analyzer',
    goal='Analyze the readability of the provided content',
    verbose=True,
    memory=True,
    backstory="You are an expert in analyzing the readability of text content using advanced readability metrics.",
    tools=[],
    allow_delegation=False
)

# Calculator Agent
calculator_agent = Agent(
    role='Calculator',
    goal='Calculate the readability percentage score of the provided content',
    verbose=True,
    memory=True,
    backstory="You excel at calculating precise readability scores for any given content.",
    tools=[CalculatorTool()],
    allow_delegation=False
)

# Academic Text Analyzer Agent
academic_analyzer = Agent(
    role='Academic Text Analyzer',
    goal='Provide detailed academic analysis of the provided content',
    verbose=True,
    memory=True,
    backstory="You specialize in analyzing text from an academic perspective, considering factors such as clarity, argument structure, and scholarly relevance.",
    tools=[],
    allow_delegation=False
)

# Marketing Copy Analyzer Agent
marketing_analyzer = Agent(
    role='Marketing Copy Analyzer',
    goal='Analyze the effectiveness of the provided marketing content',
    verbose=True,
    memory=True,
    backstory="You specialize in evaluating marketing materials, focusing on readability, persuasive language, and keyword density to enhance engagement.",
    tools=[],
    allow_delegation=False
)

# Report Compiler Agent
report_compiler = Agent(
    role='Report Compiler',
    goal='Compile the readability analysis into a comprehensive report',
    verbose=True,
    memory=True,
    backstory="You specialize in creating detailed reports that clearly present readability analysis results.",
    tools=[],
    allow_delegation=False
)

# Main Delegator Bot
delegator_bot = Agent(
    role='Delegator Bot',
    goal='Oversee the readability analysis process and delegate tasks to ensure smooth execution',
    verbose=True,
    memory=True,
    backstory="You are responsible for ensuring that each step of the readability analysis is completed accurately and efficiently.",
    tools=[],
    allow_delegation=True
)

# Readability Analysis Task
class StoringTask(Task):
    def __init__(self, database_agent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.database_agent = database_agent

    def execute(self, *args, **kwargs):
        result = super().execute(*args, **kwargs)
        key = self.description.format(**kwargs)
        self.database_agent.tools[0]._run('store', key, result)
        return result

# Define tasks
readability_task = StoringTask(
    database_agent=db_tool,
    description="Analyze the readability of the provided content: {content}.",
    expected_output='readability_report',
    agent=readability_analyzer,
)

calculation_task = StoringTask(
    database_agent=db_tool,
    description="Calculate the readability score percentage for the provided content: {content}.",
    expected_output='calculation_report',
    agent=calculator_agent,
)

academic_task = StoringTask(
    database_agent=db_tool,
    description="Provide detailed academic analysis of the provided content: {content}.",
    expected_output='academic_report',
    agent=academic_analyzer,
)

marketing_task = StoringTask(
    database_agent=db_tool,
    description="Analyze the effectiveness of the provided marketing content: {content}.",
    expected_output='marketing_report',
    agent=marketing_analyzer,
)

report_task = StoringTask(
    database_agent=db_tool,
    description="Compile the readability analysis and the calculated readability score of the provided content: {content} into a readable report.",
    expected_output='final_report',
    tools=[],
    agent=report_compiler,
    async_execution=False
)

# Streamlit UI
st.title("Readability Analysis Tool")

content = st.text_area("Provide the content you want to analyze", height=300)
include_academic_analysis = st.checkbox("Include Academic Text Analysis")
include_marketing_analysis = st.checkbox("Include Marketing Copy Analysis")

# Display "Shopping Cart" with selected metrics
st.subheader("Selected Metrics")
selected_metrics = ["Readability Analysis", "Readability Score"]
if include_academic_analysis:
    selected_metrics.append("Academic Text Analysis")
if include_marketing_analysis:
    selected_metrics.append("Marketing Copy Analysis")

st.write("You will receive the following analysis:")
for metric in selected_metrics:
    st.write(f"- {metric}")

if st.button("Analyze Readability"):
    inputs = {'content': content}

    # Prepare the agents and tasks based on user selection
    tasks = [readability_task, calculation_task, report_task]

    if include_academic_analysis:
        tasks.insert(2, academic_task)

    if include_marketing_analysis:
        tasks.insert(2, marketing_task)

    # Create Crew with selected agents and tasks
    crew = Crew(
        agents=[delegator_bot, readability_analyzer, calculator_agent, report_compiler, db_tool],
        tasks=tasks,
        process=Process.sequential
    )

    # Interpolate inputs into the agents
    for agent in crew.agents:
        agent.interpolate_inputs(inputs)

    # Interpolate inputs into the tasks
    for task in crew.tasks:
        task.description = task.description.format(content=content)

    # Execute the Crew and gather results
    print("Starting crew kickoff...")  # Debug statement
    result = crew.kickoff(inputs=inputs)
    print(f"Crew kickoff result: {result}")  # Debug statement

    # Retrieve and format the final report from the database
    readability_report = db_tool._run('retrieve', key=f"Analyze the readability of the provided content: {content}.")
    calculation_report = db_tool._run('retrieve', key=f"Calculate the readability score percentage for the provided content: {content}.")
    academic_report = db_tool._run('retrieve', key=f"Provide detailed academic analysis of the provided content: {content}.") if include_academic_analysis else ''
    marketing_report = db_tool._run('retrieve', key=f"Analyze the effectiveness of the provided marketing content: {content}.") if include_marketing_analysis else ''

    # Format the final report
    final_report = (
        f"<h2>Readability Analysis</h2>{readability_report}"
        f"<h2>Readability Score</h2>{calculation_report}"
        f"{'<h2>Academic Analysis</h2>' + academic_report if academic_report else ''}"
        f"{'<h2>Marketing Copy Analysis</h2>' + marketing_report if marketing_report else ''}"
    )

    print(f"Final report: {final_report}")  # Debug statement
    st.subheader("Readability Analysis Results")
    st.markdown(final_report, unsafe_allow_html=True)
