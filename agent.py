import asyncio
import os
from typing import TypedDict

import dotenv
from github import Auth, Github
from llama_index.core.agent import AgentWorkflow, FunctionAgent
from llama_index.core.agent.workflow import (AgentOutput, ToolCallResult,
                                             ToolCall)
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

dotenv.load_dotenv()

GITHUB_AUTH = Auth.Token(os.getenv("GITHUB_TOKEN"))
git = Github(auth=GITHUB_AUTH)

llm = OpenAI(model=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini"),
             api_key=os.getenv("OPENAI_API_KEY"),
             api_base=os.getenv("OPENAI_BASE_URL"), )

repository_id = os.getenv("REPOSITORY")
script_pr_number = os.getenv("PR_NUMBER")

repository = git.get_repo(repository_id)

CONTEXT_AGENT_PROMPT = """
You are the context gathering agent. When gathering context, you MUST gather \n: 
  - The details: author, title, body, diff_url, state, and head_sha; \n
  - Changed files; \n
  - Any requested for files; \n
Once you gather the requested info, you MUST hand control back to the Commentor Agent. 
"""

COMMENTOR_AGENT_PROMPT = """
You are the commentor agent that writes review comments for pull requests as a
human reviewer would. 
Ensure to do the following for a thorough review: 
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent. 
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing:
    - What is good about the PR?
    - Did the author follow ALL contribution rules? What is missing?
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this.
    - Are new endpoints documented? - use the diff to determine this.
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement.
 - If you need any additional details, you must hand off to the context gathering agent.
 - Once you are done drafting a review, you MUST use add_draft_comment_to_state to save the draft comment.
 - After saving the draft comment, you MUST hand off to the ReviewAndPostingAgent.
 - Do not provide the final response yourself; the ReviewAndPostingAgent is responsible for final review and posting.
 - You should directly address the author. So your comments should sound like:
 "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?"
"""
REVIEW_AGENT_PROMPT = """
You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. 
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must:
   - Be a ~200-300 word review in markdown format.
   - Specify what is good about the PR:
   - Did the author follow ALL contribution rules? What is missing?
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them?
   - Are there notes on whether new endpoints were documented?
   - Are there suggestions on which lines could be improved upon? Are these lines quoted?
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns.
 When you are satisfied, post the review to GitHub.  
 """

class PRDetails(TypedDict):
    author: str
    title: str
    body: str | None
    diff_url: str
    state: str
    head_sha: str
    commit_SHAs: list[str]


class ChangedFileDetails(TypedDict):
    filename: str
    status: str
    additions: int
    deletions: int
    changes: int
    patch: str | None


def get_pr_details(pr_number: int) -> PRDetails:
    """Fetch pull request details by number, including author, title, body, state, diff URL, and commit SHAs."""
    pull_request = repository.get_pull(pr_number)
    body = pull_request.body or "No pull request body provided."
    commit_shas = []
    commits = pull_request.get_commits()

    for commit in commits:
        commit_shas.append(commit.sha)

    return {"author": pull_request.user.login, "title": pull_request.title,
            "body": body.replace("'", ""),
            "diff_url": pull_request.diff_url,
            "state": pull_request.state, "head_sha": pull_request.head.sha,
            "commit_SHAs": commit_shas, }


def get_pr_commit_details(head_sha: str) -> list[ChangedFileDetails]:
    """Fetch changed file details for a pull request commit SHA."""
    commit = repository.get_commit(head_sha)
    changed_files = []

    for changed_file in commit.files:
        changed_files.append(
            {"filename": changed_file.filename, "status": changed_file.status,
             "additions": changed_file.additions,
             "deletions": changed_file.deletions,
             "changes": changed_file.changes, "patch": changed_file.patch, })

    return changed_files


def get_file_contents(file_path: str) -> str:
    """Fetch the contents of a file from the repository by path."""
    file_content = repository.get_contents(file_path)
    return file_content.decoded_content.decode("utf-8")

def post_final_review_comment(pr_number: int, comment: str) -> str:
    """Post a final review comment to a GitHub pull request."""
    pull_request = repository.get_pull(pr_number)
    pull_request.create_review(body=comment)
    return "Review comment posted to GitHub"


pr_details_tool = FunctionTool.from_defaults(get_pr_details)
pr_commit_details_tool = FunctionTool.from_defaults(get_pr_commit_details)
file_contents_tool = FunctionTool.from_defaults(get_file_contents)
review_comment_post_tool = FunctionTool.from_defaults(post_final_review_comment)


async def add_pr_details_to_state(ctx: Context, details: str) -> str:
    """Add gathered PR context, changed file details, and requested file contents to state"""
    async with ctx.store.edit_state() as state:
        state["state"]["gathered_contexts"] += details
        return "State updated with PR details"


async def add_draft_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """Add draft comment to state"""
    async with ctx.store.edit_state() as state:
        state["state"]["draft_comment"] = draft_comment
        return "State updated with draft comment"

async def add_final_review_to_state(ctx: Context,
                                     final_review: str) -> str:
    """Add final review comment to state"""
    async with ctx.store.edit_state() as state:
        state["state"]["final_review"] = final_review
        return "State updated with final review"


context_agent = FunctionAgent(name="ContextAgent",
                              description="Gathers context for PR review by fetching file contents, PR details, and commit details.",
                              system_prompt=CONTEXT_AGENT_PROMPT,
                              tools=[file_contents_tool, pr_details_tool,
                                     pr_commit_details_tool,
                                     add_pr_details_to_state, ],
                              can_handoff_to=["CommentorAgent"], llm=llm, )

comment_agent = FunctionAgent(name="CommentorAgent",
                              description="Writes a review comment for a PR based on gathered context.",
                              system_prompt=COMMENTOR_AGENT_PROMPT,
                              tools=[add_draft_comment_to_state],
                              can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"], llm=llm, )

review_agent = FunctionAgent(name="ReviewAndPostingAgent",
                              description=("Reviews draft comment and posts "
                                          "to GitHub when satisfied"),
                              system_prompt=REVIEW_AGENT_PROMPT,
                              tools=[review_comment_post_tool, add_final_review_to_state],
                              can_handoff_to=["CommentorAgent"], llm=llm, )

workflow_agent = AgentWorkflow(agents=[context_agent, comment_agent, review_agent],
                               root_agent=review_agent.name,
                               initial_state={"gathered_contexts": "",
                                              "draft_comment": "",
                                              "final_review": ""}, )
context = Context(workflow_agent)


async def main():
    query = f"Write a review for PR: {script_pr_number}"
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    git.close()
