import asyncio
import os
from typing import TypedDict

import dotenv
from github import Auth, Github
from llama_index.core.agent import AgentWorkflow, FunctionAgent
from llama_index.core.agent.workflow import (
    AgentOutput,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

dotenv.load_dotenv()


github_token = os.getenv("GITHUB_TOKEN")
repository_id = os.getenv("REPOSITORY")
script_pr_number = os.getenv("PR_NUMBER")
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_BASE_URL")

if not github_token:
    raise RuntimeError("Missing GITHUB_TOKEN config.")
if not repository_id:
    raise RuntimeError("Missing REPOSITORY config.")
if not script_pr_number:
    raise RuntimeError("Missing PR_NUMBER config.")

GITHUB_AUTH = Auth.Token(github_token)
git = Github(auth=GITHUB_AUTH)

llm = OpenAI(
    model=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini"),
    api_key=openai_api_key,
    api_base=openai_base_url,
)

repository = git.get_repo(repository_id)

CONTEXT_AGENT_PROMPT = """
You are the context gathering agent. When asked to gather PR review context,
call get_pr_review_context with the PR number. That tool gathers PR metadata,
the aggregate changed files, patch hunks, and CONTRIBUTING.md.
Do not call get_pr_commit_details for pull request reviews.
Once get_pr_review_context succeeds, hand control back to CommentorAgent.
Never provide a user-facing final response yourself.
"""

COMMENTOR_AGENT_PROMPT = """
You are the commentor agent that writes review comments for pull requests as a
human reviewer would.
Ensure to do the following for a thorough review:
 - Request the PR details, the aggregate changed files for the PR number,
   CONTRIBUTING.md, and other repo files you may need from the ContextAgent.
 - Once you have asked for all the needed information, write a good ~200-300
   word review in markdown format detailing:
    - What is good about the PR?
    - Did the author follow ALL contribution rules? What is missing?
    - Are there tests for new functionality? If there are new models, are
      there migrations for them? Use the diff to determine this.
    - Are new endpoints documented? - use the diff to determine this.
    - Which lines could be improved upon? Quote these lines and offer
      suggestions the author could implement.
 - If you need any additional details, you must hand off to the context
   gathering agent.
 - Once you are done drafting a review, you MUST use add_draft_comment_to_state
   to save the draft comment.
 - After saving the draft comment, you MUST hand off to ReviewAndPostingAgent.
 - Do not provide the final response yourself. If draft is not saved,
   call add_draft_comment_to_state; if it has been saved, hand off to the
   ReviewAndPostingAgent.
 - You should directly address the author. So your comments should sound like:
 "Thanks for fixing this. I think all places where we call quote should be
 fixed. Can you roll this fix out everywhere?"
"""
REVIEW_AGENT_PROMPT = """
You are the Review and Posting agent. You must use the CommentorAgent to create
a review comment. Once a review is generated, run a final check and post it to
GitHub.
   - The review must:
   - Be a ~200-300 word review in markdown format.
   - Specify what is good about the PR:
   - Did the author follow ALL contribution rules? What is missing?
   - Are there notes on test availability for new functionality? If there are
     new models, are there migrations for them?
   - Are there notes on whether new endpoints were documented?
   - Are there suggestions on which lines could be improved upon? Are these
     lines quoted?
 If the review does not meet this criteria, ask the CommentorAgent to rewrite
 and address these concerns.
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


def changed_file_to_details(changed_file) -> ChangedFileDetails:
    """Convert a PyGithub changed file object to a plain dict."""
    return {
        "filename": changed_file.filename,
        "status": changed_file.status,
        "additions": changed_file.additions,
        "deletions": changed_file.deletions,
        "changes": changed_file.changes,
        "patch": changed_file.patch,
    }


def get_pr_details(pr_number: int) -> PRDetails:
    """Fetch pull request details, including metadata and commit SHAs."""
    pull_request = repository.get_pull(pr_number)
    body = pull_request.body or "No pull request body provided."
    commit_shas = []
    commits = pull_request.get_commits()

    for commit in commits:
        commit_shas.append(commit.sha)

    return {
        "author": pull_request.user.login,
        "title": pull_request.title,
        "body": body.replace("'", ""),
        "diff_url": pull_request.diff_url,
        "state": pull_request.state,
        "head_sha": pull_request.head.sha,
        "commit_SHAs": commit_shas,
    }


def get_pr_commit_details(head_sha: str) -> list[ChangedFileDetails]:
    """Fetch changed file details for a commit SHA.

    Prefer get_pr_changed_files for pull request reviews. This fallback returns
    only files changed in one commit, not the full pull request.
    """
    if script_pr_number:
        pull_request = repository.get_pull(int(script_pr_number))
        if pull_request.head.sha == head_sha:
            return get_pr_changed_files(int(script_pr_number))

    commit = repository.get_commit(head_sha)
    changed_files = []

    for changed_file in commit.files:
        changed_files.append(changed_file_to_details(changed_file))

    return changed_files


def get_pr_changed_files(pr_number: int) -> list[ChangedFileDetails]:
    """Fetch aggregate changed file details for a pull request."""
    pull_request = repository.get_pull(pr_number)
    changed_files = []

    for changed_file in pull_request.get_files():
        changed_files.append(changed_file_to_details(changed_file))

    return changed_files


def format_changed_files(changed_files: list[ChangedFileDetails]) -> str:
    """Format changed file details compactly for the review agent state."""
    formatted_files = []

    for index, changed_file in enumerate(changed_files, start=1):
        patch = changed_file["patch"] or "No patch available."
        formatted_files.append(
            "\n".join(
                [
                    f"{index}. {changed_file['filename']}",
                    f"   Status: {changed_file['status']}",
                    f"   Additions: {changed_file['additions']}",
                    f"   Deletions: {changed_file['deletions']}",
                    "   Patch:",
                    patch,
                ]
            )
        )

    return "\n\n".join(formatted_files)


def get_pr_review_context(pr_number: int) -> str:
    """Fetch PR metadata, aggregate changed files, and contribution rules."""
    pr_details = get_pr_details(pr_number)
    changed_files = get_pr_changed_files(pr_number)
    contribution_rules = get_file_contents("CONTRIBUTING.md")

    return "\n\n".join(
        [
            "PR Details:",
            f"Author: {pr_details['author']}",
            f"Title: {pr_details['title']}",
            f"Body: {pr_details['body']}",
            f"State: {pr_details['state']}",
            f"Diff URL: {pr_details['diff_url']}",
            f"Head SHA: {pr_details['head_sha']}",
            "Changed Files:",
            format_changed_files(changed_files),
            "Contribution Rules:",
            contribution_rules,
        ]
    )


def get_file_contents(file_path: str) -> str:
    """Fetch the contents of a file from the repository by path."""
    file_content = repository.get_contents(file_path)
    return file_content.decoded_content.decode("utf-8")


def post_final_review_comment(pr_number: int, comment: str) -> str:
    """Post a final review comment to a GitHub pull request."""
    pull_request = repository.get_pull(pr_number)
    pull_request.create_review(body=comment, event="COMMENT")
    return "Review comment posted to GitHub"


pr_details_tool = FunctionTool.from_defaults(get_pr_details)
pr_commit_details_tool = FunctionTool.from_defaults(get_pr_commit_details)
pr_changed_files_tool = FunctionTool.from_defaults(get_pr_changed_files)
pr_review_context_tool = FunctionTool.from_defaults(get_pr_review_context)
file_contents_tool = FunctionTool.from_defaults(get_file_contents)
post_review_fn = post_final_review_comment
review_comment_post_tool = FunctionTool.from_defaults(post_review_fn)


async def add_pr_details_to_state(ctx: Context, details: str) -> str:
    """Add gathered PR context, changed files, and file contents to state."""
    async with ctx.store.edit_state() as state:
        state["state"]["gathered_contexts"] += f"\n{details}\n"
        return "State updated with PR details"


async def add_draft_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """Add draft comment to state"""
    async with ctx.store.edit_state() as state:
        state["state"]["draft_comment"] = draft_comment
        return "State updated with draft comment"


async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Add final review comment to state"""
    async with ctx.store.edit_state() as state:
        state["state"]["final_review"] = final_review
        return "State updated with final review"


context_agent = FunctionAgent(
    name="ContextAgent",
    description=(
        "Gathers context for PR review by fetching file contents, PR details, "
        "and commit details."
    ),
    system_prompt=CONTEXT_AGENT_PROMPT,
    tools=[
        pr_review_context_tool,
        add_pr_details_to_state,
    ],
    can_handoff_to=["CommentorAgent"],
    llm=llm,
)

comment_agent = FunctionAgent(
    name="CommentorAgent",
    description="Writes a review comment for a PR based on gathered context.",
    system_prompt=COMMENTOR_AGENT_PROMPT,
    tools=[add_draft_comment_to_state],
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
    llm=llm,
)

review_agent = FunctionAgent(
    name="ReviewAndPostingAgent",
    description=("Reviews draft comment and posts to GitHub when satisfied"),
    system_prompt=REVIEW_AGENT_PROMPT,
    tools=[review_comment_post_tool, add_final_review_to_state],
    can_handoff_to=["CommentorAgent"],
    llm=llm,
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, comment_agent, review_agent],
    root_agent=review_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "final_review": "",
    },
)
context = Context(workflow_agent)


async def main():
    query = f"Write a review for PR: {script_pr_number}"
    prompt = RichPromptTemplate(query)

    user_msg = prompt.format()
    handler = workflow_agent.run(user_msg, ctx=context, max_iterations=50)

    current_agent = None
    async for event in handler.stream_events():
        if (
            hasattr(event, "current_agent_name")
            and event.current_agent_name != current_agent
        ):
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nAgent response:", event.response.content)
            if event.tool_calls:
                tool_names = [call.tool_name for call in event.tool_calls]
                print("Selected tools: ", tool_names)
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(
                "Calling selected tool: "
                f"{event.tool_name}, with arguments: {event.tool_kwargs}"
            )

    result = await handler
    if isinstance(result, AgentOutput) and result.response.content:
        print("\\n\\nFinal response:", result.response.content)


if __name__ == "__main__":
    asyncio.run(main())
    git.close()
