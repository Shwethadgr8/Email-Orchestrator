Email Behavior Orchestrator
Overview
The Email Behavior Orchestrator is a proof-of-concept (PoC) Streamlit application designed to assist an in-house email team in managing large-scale outbound email campaigns to hotels. It processes multi-turn email threads, classifies reply behaviors (e.g., Confirmation, Objection, Escalation, New Info, Unknown), and suggests actions based on configurable rules. The system ensures the email team retains full control as decision-makers, with clear visibility into AI-driven analysis and the ability to accept or override suggestions with minimal effort.
Problem Statement
Large outbound campaigns to hotels generate diverse email replies, including confirmations, objections, escalations, new information, and unclear responses. These replies often span multiple messages within the same thread, making manual processing time-consuming. The goal is to build a system that accelerates email handling by:

Recognizing reply behaviors using AI.
Proposing appropriate actions based on configurable rules.
Keeping human responders in control to make final decisions.

Key Features

AI-Powered Behavior Classification: Uses Google Gemini (via LangChain) to analyze multi-turn email threads and classify behaviors (Confirmation, Objection, Escalation, New Info, Unknown).
Configurable Rules: Email team members can define and update behavior-to-action mappings in a rules.json file.
Human Control: Team members can accept AI-suggested actions or override them with comments, ensuring human oversight.
Persistent Decisions: Decisions (accept/override) are saved in decisions.json with timestamps and comments for auditability.
Streamlit Interface: Provides a user-friendly UI with three tabs:
Rules: Edit and save behavior-to-action rules.
Inbox: View email threads, AI classifications, suggested actions, and conversation details; accept or override actions.
Decisions: Review decision history.


Visibility: Displays AI reasoning and full conversation context for transparency.
Sample Data: Includes sample multi-turn email threads for testing.

Tech Stack

Python 3.8+
Streamlit: Web app framework for the UI.
LangChain: Integrates with Google Gemini for AI-driven email analysis.
LangGraph: Manages the workflow for classifying behaviors and suggesting actions.
JSON: Stores rules (rules.json) and decisions (decisions.json).
Dependencies: See requirements.txt for full list (streamlit, langchain-google-genai, langgraph, python-dotenv).

Setup Instructions

Clone the Repository:
git clone https://github.com/Shwethadgr8/Email-Orchestrator.git
cd Email-Orchestrator


Install Dependencies:Ensure Python 3.8+ is installed, then create a virtual environment and install requirements:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Set Up Environment Variables:Create a .env file in the root directory with your Google Gemini API key:
GEMINI_API_KEY=your_api_key_here

Obtain the API key from Google Cloud.

Run the Application:Start the Streamlit app:
streamlit run app.py

Open the provided local URL (e.g., http://localhost:8501) in your browser.


Usage

Login: Enter a username and password to access the app (basic authentication for demo purposes).
Rules Tab:
View and edit behavior-to-action mappings.
Save changes to update rules.json and refresh the workflow.


Inbox Tab:
Browse email threads with AI-classified behaviors and suggested actions.
View AI reasoning and conversation details in expanders.
Click "Accept" to confirm the AI suggestion or "Override" to provide a custom decision with a comment.


Decisions Tab:
Review the history of all decisions (accepted or overridden) with timestamps and comments.



File Structure

app.py: Main Streamlit application.
rules.json: Stores behavior-to-action rules (editable via UI).
decisions.json: Logs decisions with user, thread ID, behavior, action, and comments.
.env: Environment file for API keys (not tracked in Git).
.gitignore: Ignores .env and other sensitive files.
requirements.txt: Lists Python dependencies.

Sample Data
The app includes sample email threads (sample_threads in app.py) simulating hotel responses to campaigns. Each thread contains:

A unique thread_id.
A subject line (e.g., "Campaign: Diwali Discount Offer").
Multiple messages with sender email and content.

Notes

The .env file should not be committed to Git to protect sensitive API keys. Ensure it’s listed in .gitignore.
The app uses JSON for persistence, suitable for a PoC. For production, consider a database for scalability.
The AI model (Google Gemini) may occasionally produce unparseable responses; fallback logic ensures "Unknown" classification.

Contributing
To contribute to the project:

Fork the repository.
Create a branch: git checkout -b feature/your-feature.
Commit changes: git commit -m "Add your feature".
Push to your fork: git push origin feature/your-feature.
Open a pull request on GitHub.

To add collaborators to the repository:

Go to the repository on GitHub.
Navigate to Settings > Collaborators (under the Access section).
Click Add people, enter the GitHub username, select permissions, and send the invitation.

License
This project is unlicensed for now, as it’s a PoC. For production use, consider adding an appropriate license (e.g., MIT).
