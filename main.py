import os

from langchain.agents import Tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.utilities import GoogleSerperAPIWrapper
from crewai import Agent, Task, Process, Crew

# Set up your API key
os.environ["SERPER_API_KEY"] = "serp-api-here"

# Initialize search tool
search = GoogleSerperAPIWrapper()

search_tool = Tool(
    name="Google Search Scraper",
    func=search.run,
    description="Tool for searching the internet to find the latest information."
)

# Load additional tools
human_tools = load_tools(["human"])

# Load GPT-4 API key
api_key = os.environ.get("OPENAI_API_KEY")

# Define agents for the project

researcher = Agent(
    role="Research Specialist",
    goal="Investigate the latest applications of AI in educational apps for 2024",
    backstory="""As a Research Specialist, you excel in discovering innovative AI-driven educational apps and trends. 
    Your task is to gather data from various online sources to create a detailed report on new and emerging AI technologies 
    in the education sector. Use only the data you scrape from the internet for this report.
    """,
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
)

content_writer = Agent(
    role="Content Creator",
    goal="Draft an engaging blog post on AI trends in educational apps using simple and accessible language",
    backstory="""As a Content Creator with expertise in educational technology, your role is to simplify complex AI concepts related to educational 
    apps for a broad audience. Create a compelling blog that highlights the most recent trends and innovations in AI for education, using data from the research report.
    """,
    verbose=True,
    allow_delegation=True,
)

editor = Agent(
    role="Editorial Reviewer",
    goal="Review and provide feedback on the blog post drafts about AI in educational apps to ensure clarity and engagement",
    backstory="""As an Editorial Reviewer, you focus on enhancing the clarity, engagement, and accuracy of technical writing. 
    Provide constructive feedback to ensure the blog post is well-structured, engaging, and effectively communicates the latest trends in AI for educational apps.
    """,
    verbose=True,
    allow_delegation=True,
)

# Define tasks for the project

research_task = Task(
    description="""Gather and summarize data from the internet to create a comprehensive report on the latest AI applications in educational apps. 
    The report should be text-only and include 5-10 bullet points, each describing a specific AI-driven educational app or technology with three sentences. 
    Only use the scraped data from the internet.
    """,
    expected_output="A detailed text report with 5-10 bullet points about new AI-driven educational apps and technologies.",
    agent=researcher,
)

blog_task = Task(
    description="""Write a blog article with a catchy headline and at least 10 paragraphs summarizing the research on AI trends in educational apps. 
    The blog should be written in an engaging and simple style, accessible to a general audience. Include links to projects and tools, and format the blog in markdown as follows:
    ```
    ## [Title of post](link to project)
    - Interesting facts
    - Thoughts on the significance of the project in educational technology
    ## [Title of second post](link to project)
    - Interesting facts
    - Thoughts on the significance of the project in educational technology
    ```
    """,
    expected_output="A markdown-formatted blog article summarizing AI trends in educational apps, with a headline, at least 10 paragraphs, and links to projects.",
    agent=content_writer,
)

review_task = Task(
    description="""Ensure the blog post adheres to the following markdown format:
    ```
    ## [Title of post](link to project)
    - Interesting facts
    - Thoughts on the significance of the project in educational technology
    ## [Title of second post](link to project)
    - Interesting facts
    - Thoughts on the significance of the project in educational technology
    ```
    Review the blog for clarity, engagement, and accurate representation of AI trends in educational apps. Revise if necessary.
    """,
    expected_output="The blog post should follow the specified markdown format and effectively present AI trends in educational apps.",
    agent=editor,
)

# Instantiate the crew and start the process
crew = Crew(
    agents=[researcher, content_writer, editor],
    tasks=[research_task, blog_task, review_task],
    verbose=True,
    process=Process.sequential,  # Sequential process will execute tasks one after the other
)

# Start the crew process
result = crew.kickoff()

print("######################")
print(result)
