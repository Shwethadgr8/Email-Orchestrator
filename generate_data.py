# data.py
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
    """Main function to generate varied and realistic hotel email threads."""
    model = configure_api()
    if not model:
        return

    # --- Add lists for dynamic content generation ---
    hotel_names = [
        "The Majestic Pearl", "Sunset Vista Hotel", "Emerald Plaza", "Crimson Peaks Lodge",
        "Hotel Serenity", "The Royal Crest", "Sapphire Bay Resort", "Oakhaven Inn"
    ]
    campaign_topics = [
        "Q4 Corporate Partnership Rate", "Exclusive Holiday Event Package", "Summer Weekend Deal for Teams",
        "New Conference Venue Inquiry", "Diwali Discount Offer for Partners", "New Year Rewards Program"
    ]
    
    scenario_paths = [
        # (Your 7 scenarios remain the same)
        {"name": "Confirmation_Path", "description": "A straightforward conversation where the hotel manager quickly confirms interest and availability, leading to a successful outcome with minimal back-and-forth."},
        {"name": "Objection_Path_Pricing", "description": "The hotel manager replies that the proposed rates are too low for the season. The thread shows a negotiation attempt from the client team to address the price objection."},
        {"name": "Escalation_Path_To_Manager", "description": "The initial contact (e.g., front desk) is not authorized to handle corporate rates and escalates the request by forwarding the thread to a senior Sales Manager, who then takes over."},
        {"name": "New_Info_Path_Renovation", "description": "The hotel contact provides unexpected new information, stating that the hotel will be under renovation during the requested period and offers alternative dates or a sister property."},
        {"name": "Unknown_Case_Ambiguous_Reply", "description": "The hotel contact sends a confusing, one-line reply like 'we will look into this.' The thread shows the client team trying to get clarification on the meaning and next steps."},
        {"name": "Repeated_Outreach_Path_Ghosting", "description": "After showing initial interest, the hotel contact goes silent. The thread must show the client team sending at least two follow-up emails over several days before finally getting a response."},
        {"name": "Multi_Reply_Contact_Path_Updates", "description": "The same sales manager replies multiple times in the thread. First, they ask for details. Then, a few hours later, they reply again with another question or an update before the client has even responded to the first message."}
    ]
    
    variants_per_scenario = 1 
    total_to_generate = len(scenario_paths) * variants_per_scenario
    output_dir = "hotel_scenario_data"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving {total_to_generate} generated JSON files in '{output_dir}/' directory.")

    generated_count = 0
    for scenario in scenario_paths:
        for i in range(variants_per_scenario):
            generated_count += 1
            
            # --- MODIFIED: Select random elements for this thread ---
            selected_hotel = random.choice(hotel_names)
            selected_topic = random.choice(campaign_topics)
            hotel_email_domain = selected_hotel.lower().replace(" ", "").replace("the", "") + ".com"
            # --- END MODIFIED ---

            print(f"Generating thread {generated_count} of {total_to_generate}: {scenario['name']} for {selected_hotel}")
            
            # --- MODIFIED: The prompt is now an f-string using the dynamic variables ---
            prompt = f"""
            You are an expert synthetic data generator. Your task is to create a single, valid JSON object representing a multi-turn email thread.
            DO NOT output any text or markdown formatting before or after the JSON object.

            - The thread should be a realistic length for the scenario, anywhere from 3 to 8 emails.
            - The hotel's name for this scenario is: "{selected_hotel}".
            - The campaign topic for this scenario is: "{selected_topic}".

            Scenario to Simulate: {scenario['description']}

            Your output MUST be a JSON object with the following structure:
            {{
              "scenario_name": "{scenario['name']}",
              "scenario_description": "{scenario['description']}",
              "thread": [
                {{
                  "message_id": "msg_01",
                  "reply_to_message_id": null,
                  "day": 0,
                  "time": "09:15",
                  "sender": "Your Email Reply Team",
                  "recipient": "contact@{hotel_email_domain}",
                  "cc": [],
                  "subject": "Re: {selected_topic}",
                  "body": "Hi team, following up on our {selected_topic} campaign. Are you the right contact for this?"
                }}
              ]
            }}
            Generate the complete JSON object for the scenario now.
            """
            
            try:
                # (The rest of the try/except block remains the same)
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