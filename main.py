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
load_dotenv()

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

# ---- Sample Threads with Multi-Turn Messages ----
sample_threads = [
    {
        "thread_id": "C1",
        "subject": "Campaign: Diwali Discount Offer",
        "messages": [
            {"content": "hotel_abc@hotel.com: Hello, we received your campaign email."},
            {"content": "hotel_abc@hotel.com: We are interested, can you share more details?"},
            {"content": "hotel_abc@hotel.com: Also, we need clarification on commission rates."},
            {"content": "hotel_abc@hotel.com: Thanks, we will confirm by tomorrow."}
        ]
    },
    {
        "thread_id": "C2",
        "subject": "Campaign: New Year Rewards Program",
        "messages": [
            {"content": "hotel_xyz@hotel.com: We are not happy with the previous rewards."},
            {"content": "hotel_xyz@hotel.com: Please escalate to your manager immediately!"},
            {"content": "hotel_xyz@hotel.com: We changed our contact person, please update."}
        ]
    },
    {
        "thread_id": "C3",
        "subject": "Campaign: Summer Festival Offer",
        "messages": [
            {"content": "hotel_pqr@hotel.com: Yes, count us in for the campaign!"},
            {"content": "hotel_pqr@hotel.com: Thanks for the update."}
        ]
    }
]

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
    prompt = f"""
You are a campaign assistant analyzing hotel email conversations.
Each message is part of a conversation with a hotel.

Step 1: Read each message in order.
Step 2: Identify if the message indicates any of these behaviors:
  - Confirmation
  - Objection
  - Escalation
  - New Info
  - Unknown

Step 3: After reading all messages, determine the **overall classification** of the conversation.
Step 4: Provide a **brief reasoning**.

Conversation:
{all_text}

Output format (JSON ONLY):
{{
  "behavior": "<one of Confirmation, Objection, Escalation, New Info, Unknown>",
  "reason": "<brief explanation>"
}}
"""
    resp = llm.invoke(prompt)
    parsed = parse_model_response(resp.content)
    state["behavior"] = parsed.get("behavior", "Unknown")
    state["reason"] = parsed.get("reason", "")
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
else:
    tab1, tab2, tab3 = st.tabs(["Rules", "Inbox", "Decisions"])

    # ---- Rules Tab ----
    with tab1:
        st.header("Edit Rules")
        rules = load_rules()  # Load current rules
        new_rules = {}
        for k, v in rules.items():
            new_rules[k] = st.text_input(f"{k}", v)
        if st.button("Save Rules"):
            with open(RULES_FILE, "w") as f:
                json.dump(new_rules, f, indent=2)
            st.session_state.rules_version += 1  # Increment to invalidate cache
            st.cache_data.clear()  # Clear cache to force workflow reload
            st.session_state.rules_saved = True  # Set saved state
            st.rerun()

        # Show persistent "Saved" message
        if st.session_state.rules_saved:
            st.success("Rules saved!", icon="✅")
            if st.button("Clear Saved Message"):
                st.session_state.rules_saved = False
                st.rerun()

    # ---- Inbox Tab ----
    with tab2:
        st.header("Campaign Replies Inbox")
        inbox_data = []
        for thread in sample_threads:
            messages_for_thread = [HumanMessage(content=m["content"]) for m in thread["messages"]]
            print(f"Messages for {thread['thread_id']}: {[m.content for m in messages_for_thread]}")
            state = run_workflow(messages_for_thread, thread["thread_id"], st.session_state.rules_version)
            print(f"Thread {thread['thread_id']} - Final State: {state}")
            inbox_data.append({
                "thread_id": thread["thread_id"],
                "hotels": ", ".join(set(m["content"].split(":")[0] for m in thread["messages"])),
                "behavior": state.get("behavior", "Unknown"),
                "rule": state.get("suggestion", "Review manually"),
                "reason": state.get("reason", ""),
                "messages": thread["messages"]
            })
        print("Inbox Data:", inbox_data)

        for row in inbox_data:
            cols = st.columns([2, 2, 2, 2, 2])
            cols[0].markdown(f"**{row['hotels']}**")
            cols[1].markdown(f"**{row['behavior']}**")
            cols[2].markdown(row['rule'])
            key_accept = f"accept_{row['thread_id']}"
            key_override = f"override_{row['thread_id']}"
            decision_key = f"decision_{row['thread_id']}"
            comment_key = f"comment_{row['thread_id']}"

            if cols[3].button("Accept", key=key_accept):
                save_decision(
                    st.session_state.username,
                    row["thread_id"],
                    row["behavior"],
                    row["rule"],
                    "Accepted"
                )
                st.session_state[decision_key] = "Accepted"
                st.rerun()

            if cols[4].button("Override", key=key_override):
                st.session_state[decision_key] = "Overridden"
                st.rerun()

            decision = st.session_state.get(decision_key)
            if decision == "Accepted":
                cols[3].success("Accepted", icon="✅")
            elif decision == "Overridden":
                cols[4].warning("Overridden", icon="⚠️")
                comment = st.text_input("Reason for override:", key=comment_key)
                # Show "Save Override" only if not saved
                if not st.session_state.get(f"saved_{row['thread_id']}"):
                    if st.button("Save Override", key=f"save_override_{row['thread_id']}"):
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

            # Show saved message and Clear button
            if st.session_state.get(f"saved_{row['thread_id']}"):
                cols[4].success("Saved", icon="✅")
                if st.button("Clear", key=f"clear_{row['thread_id']}"):
                    del st.session_state[f"saved_{row['thread_id']}"]
                    st.rerun()

            with st.expander("View Conversation"):
                st.markdown(f"**AI Analysis**: {row['reason']}")
                for msg in row["messages"]:
                    st.write(msg["content"])

    # ---- Decisions Tab ----
    with tab3:
        st.header("Decision History")
        try:
            with open(DECISIONS_FILE, "r") as f:
                try:
                    decisions = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: {DECISIONS_FILE} is invalid or empty. Displaying empty list.")
                    decisions = []
        except FileNotFoundError:
            print(f"Warning: {DECISIONS_FILE} not found. Displaying empty list.")
            decisions = []
        for d in decisions:
            st.write(d)