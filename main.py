import streamlit as st
import json
import os
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AnyMessage
from typing import TypedDict, Annotated
import operator
import re
from langgraph.graph import StateGraph
from dotenv import load_dotenv
import pandas as pd 
import plotly.express as px 


load_dotenv()

# ---- Feedback and Comment Storage ----
FEEDBACK_FILE = "feedback.json"
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump([], f)

def store_feedback(thread_id, ai_response, user_feedback, comment, overridden=False):
    try:
        with open(FEEDBACK_FILE, "r") as f:
            feedback_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        feedback_data = []

    entry = {
        "thread_id": thread_id,
        "ai_response": ai_response,
        "user_feedback": user_feedback,
        "comment": comment,
        "overridden": overridden,
        "timestamp": datetime.now().isoformat()
    }

    # Check if feedback for this thread already exists
    for i, fb in enumerate(feedback_data):
        if fb["thread_id"] == thread_id:
            feedback_data[i] = entry
            break
    else:
        feedback_data.append(entry)

    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback_data, f, indent=2)

# ---- Draft Email Suggestion Helper ----
def suggest_draft_email(conversation, behavior, tone="auto"):
    """
    Suggest a polite, context-appropriate draft email based on conversation and behavior.
    """
    if tone == "auto":
        if behavior in ["Confirmation", "Escalation"]:
            tone = "professional"
        elif behavior in ["Objection", "New Info", "Action Required"]:
            tone = "concise"
        else:
            tone = "friendly"

    # Simple template logic (can be replaced with LLM call for more advanced suggestions)
    if behavior == "Confirmation":
        draft = "Thank you for your confirmation. We look forward to working with you. Please let us know if you need any further information."
    elif behavior == "Objection":
        draft = "Thank you for sharing your concerns. We appreciate your feedback and would be happy to discuss possible solutions or adjustments. Please let us know how we can address your objections."
    elif behavior == "Escalation":
        draft = "Thank you for escalating this matter. We will review the details and get back to you as soon as possible. If you have any additional information, please share it with us."
    elif behavior == "New Info":
        draft = "Thank you for providing this new information. We will update our records accordingly. Please let us know if there is anything else we should be aware of."
    elif behavior == "Action Required":
        draft = "Thank you for your message. We will take the necessary action and update you soon. If you have any further questions, please let us know."
    elif behavior == "Awaiting Reply":
        draft = "Just following up on our previous conversation. Please let us know if you have any updates or questions."
    else:
        draft = "Thank you for your message. Please let us know how we can assist you further."

    # Add polite closing
    if tone == "professional":
        closing = "Best regards,\nYour Email Reply Team"
    elif tone == "concise":
        closing = "Regards,\nYour Email Reply Team"
    else:
        closing = "Thanks!\nYour Email Reply Team"

    return f"{draft}\n\n{closing}"

# ---- Define Custom State Schema ----
class CustomState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    behavior: str
    reason: str
    suggestion: str

# ---- Load Rules ----
RULES_FILE = "rules.json"
if not os.path.exists(RULES_FILE):
    default_rules = {
        "Confirmation": "Mark as confirmed in CRM",
        "Objection": "Forward to Sales Team",
        "Escalation": "Escalate to Support Desk",
        "New Info": "Update CRM with new information",
        "Unknown": "Review manually"
    }
    with open(RULES_FILE, "w") as f:
        json.dump(default_rules, f, indent=2)

def load_rules():
    try:
        with open(RULES_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: Could not load {RULES_FILE}. Using default rules.")
        return {
            "Confirmation": "Mark as confirmed in CRM",
            "Objection": "Forward to Sales Team",
            "Escalation": "Escalate to Support Desk",
            "New Info": "Update CRM with new information",
            "Unknown": "Review manually"
        }

# ---- Decisions File ----
DECISIONS_FILE = "decisions.json"
if not os.path.exists(DECISIONS_FILE):
    with open(DECISIONS_FILE, "w") as f:
        json.dump([], f)
        
        
# ---- Helper Function to Load Real Data ----
def load_threads_from_directory(directory_path):
    """Loads all thread JSON files from a directory and transforms them."""
    threads = []
    # Check if the directory exists
    if not os.path.exists(directory_path):
        # Use st.error to display a clear error in the app if the path is wrong
        st.error(f"Data directory not found: {directory_path}")
        return []

    # Sort the files to ensure a consistent order
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".json"):
            filepath = os.path.join(directory_path, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)

                # This part transforms your file's structure into the format the app expects
                transformed_thread = {
                    "thread_id": data.get("scenario_name", filename),
                    "subject": data.get("thread", [{}])[0].get("subject", "No Subject"),
                    "messages": [
                        # We combine 'sender' and 'body' into a single 'content' string
                        {"content": f"{msg.get('sender', 'Unknown')}: {msg.get('body', '')}"}
                        for msg in data.get("thread", [])
                    ]
                }
                threads.append(transformed_thread)
            except (json.JSONDecodeError, IndexError) as e:
                # This will print a warning in your terminal if a file is broken
                print(f"Warning: Could not load or parse {filename}. Error: {e}")
    return threads

# ---- Sample Threads with Multi-Turn Messages ----
DATA_DIRECTORY="hotel_scenario_data"
sample_threads = load_threads_from_directory(DATA_DIRECTORY)

# ---- LangGraph Setup ----
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"))

# Parse LLM response safely
def parse_model_response(response_text):
    cleaned = re.sub(r"^```json|```$", "", response_text.strip(), flags=re.MULTILINE)
    print(f"Raw LLM Response: {response_text}")
    print(f"Cleaned Response: {cleaned}")
    try:
        parsed = json.loads(cleaned)
        print(f"Parsed LLM Response: {parsed}")
        return parsed
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {cleaned}, Error: {e}")
        return {"behavior": "Unknown", "reason": "Failed to parse model response."}

def classify_behavior(state: CustomState):
    all_text = "\n".join([m["content"] if isinstance(m, dict) else m.content for m in state["messages"]])
# (Inside your classify_behavior function)

    prompt = f"""
    You are an expert email thread analyst. Your goal is to determine the primary business outcome or state of a conversation by following the user's instructions precisely.

    Follow these steps:
    1.  **Analyze the Theme:** Read the entire conversation and identify its main purpose (e.g., negotiation, inquiry, closing a deal).
    2.  **Determine the Outcome:** What is the most significant conclusion or current state of the thread? Prioritize the main business outcome over simple turn-taking.
    3.  **Classify:** Based on this analysis, choose the BEST single category from the list below:
        - **Confirmation:** The hotel has clearly and definitively agreed to the deal. The negotiation is successfully over.
        - **Objection:** The hotel has raised a significant problem (like price or availability) that is currently blocking the deal.
        - **Escalation:** The primary contact has changed because the conversation was passed to a person with more authority.
        - **New Info:** The hotel has provided new, unexpected information that changes the context of the deal (e.g., renovations).
        - **Awaiting Reply:** We are waiting for a response from the hotel to a question or proposal we have sent. (Only use this if no other major outcome fits).
        - **Action Required:** The hotel has asked us a direct question and is waiting for our response. (Only use this if no other major outcome fits).
        - **Unknown:** The intent of the conversation is genuinely unclear.
    4.  **Reason:** Provide a concise, one-sentence reason for your classification.

    ---
    **Here is a perfect example of how to do this:**

    Conversation History:
    "Your Email Reply Team: Hi, we'd like to book a block of 20 rooms for our corporate event from Oct 10-12 at your proposed rate of $150/night.
    sales@hotel.com: Thanks for reaching out. Unfortunately, due to a city-wide conference that week, our rate for those dates is firm at $220/night."

    Your JSON Output:
    {{
      "behavior": "Objection",
      "reason": "The hotel has objected to the proposed price and countered with a higher rate."
    }}
    ---

    Now, analyze the following real conversation and provide the output in the same JSON format.

    **Real Conversation History:**
    {all_text}

    **Output JSON ONLY:**
    """
    resp = llm.invoke(prompt)
    parsed = parse_model_response(resp.content)
    
    # This block makes the function robust against unexpected list outputs from the LLM.
    final_parsed_dict = {}
    if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
        # If the LLM returned a list of dictionaries, we'll just take the first one.
        final_parsed_dict = parsed[0]
    elif isinstance(parsed, dict):
        # If it's a dictionary as expected, we use it directly.
        final_parsed_dict = parsed
    # If it's an empty list or another unexpected type, final_parsed_dict will remain empty.
    

    # Now we can safely use .get() on a dictionary.
    state["behavior"] = final_parsed_dict.get("behavior", "Unknown")
    state["reason"] = final_parsed_dict.get("reason", "Model returned an unexpected format.")
    print(f"ClassifyBehavior - State: {state}")
    return state

def suggest_action(state: CustomState):
    rules = load_rules()  # Load rules dynamically
    behavior = state.get("behavior", "Unknown")
    print(f"SuggestAction - Input State: {state}")
    print(f"SuggestAction - Behavior: {behavior}")
    action = rules.get(behavior, "Review manually")
    state["suggestion"] = action
    print(f"SuggestAction - Output State: {state}")
    return state

# Set up LangGraph with CustomState
graph = StateGraph(CustomState)
graph.add_node("ClassifyBehavior", classify_behavior)
graph.add_node("SuggestAction", suggest_action)
graph.add_edge("ClassifyBehavior", "SuggestAction")
graph.set_entry_point("ClassifyBehavior")
workflow = graph.compile()

# Workflow function with caching
@st.cache_data(show_spinner=False)
def run_workflow(_messages, thread_id, _rules_version=0):
    state = CustomState(messages=_messages, behavior="", reason="", suggestion="")
    result = workflow.invoke(state)
    print(f"Workflow Result for {thread_id}: {result}")
    return result

# Save decisions (updated to handle single entry per thread and comments for overrides)
def save_decision(user, thread_id, behavior, suggestion, decision, comment=None):
    try:
        with open(DECISIONS_FILE, "r") as f:
            try:
                decisions = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {DECISIONS_FILE} is invalid or empty. Initializing with empty list.")
                decisions = []
    except FileNotFoundError:
        print(f"Warning: {DECISIONS_FILE} not found. Initializing with empty list.")
        decisions = []

    # Check if an entry for thread_id already exists
    for i, entry in enumerate(decisions):
        if entry["thread_id"] == thread_id:
            # Update existing entry
            decisions[i] = {
                "user": user,
                "thread_id": thread_id,
                "behavior": behavior,
                "suggestion": suggestion,
                "decision": decision,
                "comment": comment,
                "timestamp": datetime.now().isoformat()
            }
            break
    else:
        # Append new entry if no existing entry found
        decisions.append({
            "user": user,
            "thread_id": thread_id,
            "behavior": behavior,
            "suggestion": suggestion,
            "decision": decision,
            "comment": comment,
            "timestamp": datetime.now().isoformat()
        })

    with open(DECISIONS_FILE, "w") as f:
        json.dump(decisions, f, indent=2)

# ---- Streamlit App ----
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "rules_version" not in st.session_state:
    st.session_state.rules_version = 0
if "rules_saved" not in st.session_state:
    st.session_state.rules_saved = False

if not st.session_state.logged_in:
    st.title("Campaign Reply Classifier - Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username and password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Enter username and password")
# (This code replaces the entire 'else' block after the login)
else:
    # ---- Sidebar for Rules ----
    with st.sidebar:
        st.header("📝 Edit Rules")
        st.markdown("Define the default action for each detected behavior.")
        
        rules = load_rules()
        new_rules = {}
        for behavior, action in rules.items():
            # Using a more descriptive label for clarity
            new_rules[behavior] = st.text_input(f"Action for '{behavior}'", action, key=f"rule_{behavior}")
        
        if st.button("Save Rules"):
            with open(RULES_FILE, "w") as f:
                json.dump(new_rules, f, indent=2)
            st.session_state.rules_version += 1
            st.cache_data.clear()
            st.session_state.rules_saved = True
            st.rerun()

        if st.session_state.get("rules_saved", False):
            st.success("Rules saved!", icon="✅")
            
    st.markdown("---") # Adds a visual separator
    st.header("🗂️ Inbox Filter")

    # Get the list of possible behaviors from your rules file
    rules = load_rules()
    filter_options = ["All"] + list(rules.keys()) 

    # Create the selectbox widget
    st.selectbox(
        "Show threads with behavior:",
        options=filter_options,
        key="inbox_filter"  # We give it a key to access its value later
)

    # ---- Main Content Area with Tabs ----
    tab_inbox, tab_decisions = st.tabs(["Inbox", "Decisions"])

    # ---- Inbox Tab ----
    with tab_inbox:
        st.header("📨 Campaign Replies Inbox")
        # (The rest of your inbox code goes here, unchanged)
        inbox_data = []
        for thread in sample_threads:
            messages_for_thread = [HumanMessage(content=m["content"]) for m in thread["messages"]]
            state = run_workflow(messages_for_thread, thread["thread_id"], st.session_state.rules_version)
            inbox_data.append({
                "thread_id": thread["thread_id"],
                "hotels": ", ".join(set(m["content"].split(":")[0] for m in thread["messages"])),
                "behavior": state.get("behavior", "Unknown"),
                "rule": state.get("suggestion", "Review manually"),
                "reason": state.get("reason", ""),
                "messages": thread["messages"]
            })
            
        if st.session_state.inbox_filter == "All":
            filtered_inbox_data = inbox_data
        else:
            filtered_inbox_data = [
                row for row in inbox_data 
                if row['behavior'] == st.session_state.inbox_filter
        ]
        
        if not filtered_inbox_data:
            st.info("No threads match the current filter.")
        else:
            for row in filtered_inbox_data:
                # Create a container for each thread to act as a "card"
                with st.container(border=True):
                    # --- ROW 1: Key Information ---
                    col1, col2, col3 = st.columns([2, 1, 2])
                    with col1:
                        # Clean up the participant list for readability
                        participants = row['hotels'].replace(", Your Email Reply Team", "").strip()
                        st.markdown("**Conversation with:**")
                        st.caption(participants)
                    with col2:
                        # Create colored "badges" for the behavior classification
                        behavior = row['behavior']
                        if behavior == "Confirmation":
                            badge_color = "green"
                        elif behavior in ["Objection", "Escalation"]:
                            badge_color = "red"
                        elif behavior == "Action Required":
                            badge_color = "orange"
                        else:  # New Info, Awaiting Reply, Unknown
                            badge_color = "blue"
                        
                        # Using markdown for a simple, colored badge
                        st.markdown("**Behavior:**")
                        st.markdown(f"<span style='color:{badge_color};'>●</span> {behavior}", unsafe_allow_html=True)
                    with col3:
                        st.markdown("**Suggested Action:**")
                        st.caption(row['rule'])
                    
                    # --- ROW 2: Conversation Expander ---
                    with st.expander("View Conversation"):
                        st.markdown(f"**AI Analysis**: {row['reason']}")
                        st.markdown("---")
                        for msg in row["messages"]:
                            content = msg.get("content", ":")
                            parts = content.split(":", 1)
                            sender = parts[0].strip()
                            body = parts[1].strip() if len(parts) > 1 else ""
                            role = "assistant" if "Your Email Reply Team" in sender else "user"
                            with st.chat_message(role):
                                st.markdown(f"**From:** {sender}")
                                st.write(body)

                    # --- ROW 2.5: Editable Draft Email Suggestion ---
                    st.markdown("**Draft Email Suggestion:**")
                    draft_key = f"draft_{row['thread_id']}"
                    default_draft = suggest_draft_email(row["messages"], row["behavior"])
                    draft_text = st.text_area(
                        "Edit the draft before sending:",
                        value=default_draft,
                        key=draft_key,
                        height=120,
                        label_visibility="collapsed"
                    )
                    if st.button("Save Feedback & Draft", key=f"save_feedback_{row['thread_id']}"):
                        # Store feedback and draft for fine-tuning
                        store_feedback(
                            row["thread_id"],
                            ai_response=row["reason"],
                            user_feedback=draft_text,
                            comment=st.session_state.get(f"comment_{row['thread_id']}", ""),
                            overridden=st.session_state.get(f"saved_{row['thread_id']}", False)
                        )
                        st.success("Feedback and draft saved for this thread.")
                    
                    # --- ROW 3: Action Buttons and Status ---
                    key_accept = f"accept_{row['thread_id']}"
                    key_override = f"override_{row['thread_id']}"
                    decision_key = f"decision_{row['thread_id']}"
                    comment_key = f"comment_{row['thread_id']}"
                    
                    b_col1, b_col2, b_col3 = st.columns([1, 1, 3])
                    
                    decision = st.session_state.get(decision_key)
                    
                    with b_col1:
                        if decision == "Accepted":
                            st.success("Accepted", icon="✅")
                        else:
                            if st.button("Accept", key=key_accept, use_container_width=True):
                                save_decision(
                                    st.session_state.username,
                                    row["thread_id"],
                                    row["behavior"],
                                    row["rule"],
                                    "Accepted"
                                )
                                st.session_state[decision_key] = "Accepted"
                                st.rerun()
                    
                    with b_col2:
                        if decision == "Overridden":
                            if not st.session_state.get(f"saved_{row['thread_id']}"):
                                st.warning("Overridden", icon="⚠️")
                            else:
                                st.success("Saved", icon="✅")
                        else:
                            if st.button("Override", key=key_override, use_container_width=True):
                                st.session_state[decision_key] = "Overridden"
                                st.rerun()
                    
                    with b_col3:
                        # Handle override comment and save logic
                        if decision == "Overridden" and not st.session_state.get(f"saved_{row['thread_id']}"):
                            comment_col, save_col, clear_col = st.columns([2, 1, 1])
                            with comment_col:
                                comment = st.text_input("Reason for override:", key=comment_key, label_visibility="collapsed")
                            with save_col:
                                if st.button("Save Override", key=f"save_override_{row['thread_id']}", use_container_width=True):
                                    save_decision(
                                        st.session_state.username,
                                        row["thread_id"],
                                        row["behavior"],
                                        row["rule"],
                                        "Overridden",
                                        comment
                                    )
                                    st.session_state[f"saved_{row['thread_id']}"] = True
                                    st.success("Override saved!")
                                    st.rerun()
                        
                        # Clear button for saved decisions
                        if st.session_state.get(f"saved_{row['thread_id']}"):
                            if st.button("Clear", key=f"clear_{row['thread_id']}"):
                                del st.session_state[f"saved_{row['thread_id']}"]
                                st.rerun()

        # ---- Feedback Tab (Optional) ----
        # You can add a feedback review tab if desired
        # with st.expander("View Feedback Log"):
        #     try:
        #         with open(FEEDBACK_FILE, "r") as f:
        #             feedback_data = json.load(f)
        #     except (FileNotFoundError, json.JSONDecodeError):
        #         feedback_data = []
        #     st.json(feedback_data)

        # ---- Decisions Tab ----
        with tab_decisions:
            st.header("📊 Decision History & Analytics")

            try:
                with open(DECISIONS_FILE, "r") as f:
                    decisions_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                decisions_data = []

            if not decisions_data:
                st.info("No decision history has been recorded yet.")
            else:
                # --- Analytics Section ---
                df = pd.DataFrame(decisions_data)
                
                # 1. AI Accuracy Calculation
                total_decisions = len(df)
                overridden_decisions = len(df[df['decision'] == 'Overridden'])
                accuracy = ((total_decisions - overridden_decisions) / total_decisions) * 100 if total_decisions > 0 else 100

                # 2. Behavior Distribution
                behavior_counts = df['behavior'].value_counts().reset_index()
                behavior_counts.columns = ['behavior', 'count']

                # Display Metrics and Charts
                st.subheader("Key Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="AI Suggestion Accuracy",
                        value=f"{accuracy:.1f}%",
                        help="The percentage of AI suggestions that were 'Accepted' by a human."
                    )
                with col2:
                    st.metric(
                        label="Total Decisions Logged",
                        value=total_decisions
                    )
                
                st.subheader("Behavior Classification Breakdown")
                fig = px.pie(
                    behavior_counts,
                    names='behavior',
                    values='count',
                    title='Distribution of AI-Classified Behaviors',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")

                # --- Raw Data and Download Section ---
                st.subheader("📑 Raw Decision Log")

                # Download Button
                with open(DECISIONS_FILE, "r") as f:
                    st.download_button(
                        label="Download Decisions as JSON",
                        data=f.read(),
                        file_name="decisions.json",
                        mime="application/json"
                    )

                # Display the raw data in an expander
                with st.expander("View Raw JSON Data"):
                    st.json(decisions_data)
                            