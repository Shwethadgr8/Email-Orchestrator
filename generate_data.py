import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import random

def configure_api():
    """Loads environment variables and configures the Gemini API."""
    load_dotenv()
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file or environment variables.")
        genai.configure(api_key=api_key)
        print("API configured successfully.")
        # Increased temperature for more creative/varied outputs
        return genai.GenerativeModel('gemini-1.5-flash-latest', generation_config={"temperature": 0.8})
    except Exception as e:
        print(f"Configuration Error: {e}")
        return None

def main():
    """Main function to generate varied, realistic, and labeled hotel email threads."""
    model = configure_api()
    if not model:
        return

    # --- Lists for dynamic content generation ---
    hotel_names = [
        "The Majestic Pearl", "Sunset Vista Hotel", "Emerald Plaza", "Crimson Peaks Lodge",
        "Hotel Serenity", "The Royal Crest", "Sapphire Bay Resort", "Oakhaven Inn"
    ]
    campaign_topics = [
        "Q4 Corporate Partnership Rate", "Exclusive Holiday Event Package", "Summer Weekend Deal for Teams",
        "New Conference Venue Inquiry", "Diwali Discount Offer for Partners", "New Year Rewards Program"
    ]
    
    scenario_paths = [
        {"name": "Confirmation_Path", "description": "A straightforward conversation where the hotel manager quickly confirms interest and availability, leading to a successful outcome with minimal back-and-forth."},
        {"name": "Objection_Path_Pricing", "description": "The hotel manager replies that the proposed rates are too low for the season. The thread shows a negotiation attempt from the client team to address the price objection."},
        {"name": "Escalation_Path_To_Manager", "description": "The initial contact (e.g., front desk) is not authorized to handle corporate rates and escalates the request by forwarding the thread to a senior Sales Manager, who then takes over."},
        {"name": "New_Info_Path_Renovation", "description": "The hotel contact provides unexpected new information, stating that the hotel will be under renovation during the requested period and offers alternative dates or a sister property."},
        {"name": "Unknown_Case_Ambiguous_Reply", "description": "The hotel contact sends a confusing, one-line reply like 'we will look into this.' The thread shows the client team trying to get clarification on the meaning and next steps."},
        {"name": "Repeated_Outreach_Path_Ghosting", "description": "After showing initial interest, the hotel contact goes silent. The thread must show the client team sending at least two follow-up emails over several days before finally getting a response."},
        {"name": "Multi_Reply_Contact_Path_Updates", "description": "The same sales manager replies multiple times in the thread. First, they ask for details. Then, a few hours later, they reply again with another question or an update before the client has even responded to the first message."}
    ]
    
    variants_per_scenario = 2
    total_to_generate = len(scenario_paths) * variants_per_scenario
    output_dir = "hotel_scenario_data"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving {total_to_generate} generated JSON files in '{output_dir}/' directory.")

    generated_count = 0
    for scenario in scenario_paths:
        for i in range(variants_per_scenario):
            generated_count += 1
            selected_hotel = random.choice(hotel_names)
            selected_topic = random.choice(campaign_topics)
            hotel_email_domain = selected_hotel.lower().replace(" ", "").replace("the", "") + ".com"

            print(f"Generating thread {generated_count} of {total_to_generate}: {scenario['name']} for {selected_hotel}")
            
            # --- NEW ADVANCED PROMPT ---
            prompt = f"""
            You are an expert synthetic data generator. Your task is to create a single, valid JSON object representing a realistic, complex, multi-turn email thread.
            DO NOT output any text or markdown formatting before or after the JSON object.

            **Core Instructions:**
            1.  **Behavior Label:** The root of the JSON must include a "behavior_label" field with the value: "{scenario['name']}". This is the ground truth.
            2.  **Realism:** Emails must have a professional tone and include realistic details like professional signatures (e.g., "John Doe, Sales Manager, [Hotel Name]"). Mention hotel-specific details (e.g., room types, amenities, conference hall names) where appropriate for "{selected_hotel}".
            3.  **Timestamps:** Ensure timestamps are realistic and incremental, showing logical delays between replies (minutes for quick replies, hours or days for considered responses).
            4.  **Complexity:** The conversation should realistically reflect the scenario: "{scenario['description']}". Use "reply_to_message_id" to show complex threading (e.g., someone replying to an older message). Use the "cc" field where logical (e.g., when adding a manager).
            5.  **Dynamic Initial Email:** The first message should be a realistic initial outreach for the campaign topic: "{selected_topic}".

            **Output JSON Structure:**
            {{
              "behavior_label": "{scenario['name']}",
              "thread": [
                {{
                  "message_id": "msg_01",
                  "reply_to_message_id": null,
                  "day": 0,
                  "time": "HH:MM",
                  "sender": "Your Email Reply Team",
                  "recipient": "contact@{hotel_email_domain}",
                  "cc": [],
                  "subject": "Re: {selected_topic}",
                  "body": "(Your generated realistic email body, including signature)"
                }}
              ]
            }}

            Generate the complete JSON object for the scenario now.
            """
            
            try:
                response = model.generate_content(prompt)
                clean_response_text = response.text.strip().replace("```json", "").replace("```", "")
                json_data = json.loads(clean_response_text)
                file_name = f"{scenario['name']}.json"
                output_path = os.path.join(output_dir, file_name)
                
                with open(output_path, 'w') as f:
                    json.dump(json_data, f, indent=4)
                    
            except (json.JSONDecodeError, Exception) as e:
                print(f"  !! An error occurred while processing {scenario['name']}: {e}")

    print(f"\nData generation complete. {generated_count} files created in '{output_dir}/'.")

if __name__ == "__main__":
    main()