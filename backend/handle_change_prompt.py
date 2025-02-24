from chat_bot.bot_choser import get_gpt_response
from chat_bot.gpt_for_everyone import fetch_gpt_response
from chat_bot.gpt_bot import build_pinecone_information_prompt
from chat_bot.gpt_bot import append_reply_to_chat_history, fetch_paid_openai_response
from utils.dict_lists import keys
from pre_process_hard_filters import get_outfit_selected
from main import OUTFIT_HISTORY, user_purchase_csv
from openai import OpenAI

client = OpenAI()

def get_gpt_response(prompt, paid=True):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting GPT response: {e}")
        return None

def handle_change_prompt(outfit):
    from utils.process_outfit import extract_category_info
    category_info = extract_category_info(outfit)
    pinecone_prompt = build_pinecone_information_prompt(category_info)
    # append_reply_to_chat_history(pinecone_prompt)

    # response = fetch_paid_openai_response(pinecone_prompt)
    # response = fetch_gpt_response(pinecone_prompt)
    response = get_gpt_response(pinecone_prompt,paid=True)
    if response is None:
        print("Sever is down")
        return

    print(response)

def handle_next_prompt(user_prompt: str, prev_outfit_index):

    # next_response = fetch_paid_openai_response(user_prompt)
    # next_response = fetch_gpt_response(user_prompt)
    next_response = get_gpt_response(user_prompt,paid=True)

    from prompt_insights import parse_text
    if next_response is None:
        print("Sever is down")
        return

    curr_categories = []
    category_dict_array = parse_text(next_response, keys=keys)

    print("\n category_dict_array:", category_dict_array)

    for category_dict in category_dict_array:
        category = category_dict['category']
        to_change = category_dict.get('to_change', False)
        if to_change:
            curr_categories.append(category)
        else:
            curr_categories.append('none')


    # each category is a dictionary that contain a key called to_change
    # add it to curr_categories, if not present add 'none'
    outfit = get_outfit_selected(user_prompt, user_purchase_csv, curr_categories, category_dict_array)

    # loading last articles from history.
    for i, article_dict in enumerate(outfit):
        if article_dict is None:
            # load from history
            article_dict = OUTFIT_HISTORY[prev_outfit_index]['outfit'][i]
            outfit[i] = article_dict

    return outfit
