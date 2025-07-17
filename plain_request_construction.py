import praw
import csv
import time
from typing import Optional, List, Dict
import pandas as pd
import openai
import os
import json
import dotenv

dotenv.load_dotenv()

def fetch_reddit_posts_to_csv(subreddit_name: str, keyword: str, 
                              client_id: str, client_secret: str, 
                              user_agent: str, password: str, limit: int = 150) -> None:
    """
    Fetches top text posts from past year in a subreddit containing a keyword and saves to CSV.
    
    Args:
        subreddit_name (str): Name of the subreddit (without 'r/')
        keyword (str): Keyword to search for in posts
        client_id (str): Reddit API client ID
        client_secret (str): Reddit API client secret
        user_agent (str): User agent string for API requests
        limit (int): Number of posts to fetch (default: 150)
    
    Note: Only includes text posts (not link posts) from the past year, sorted by top score.
    """
    
    # Initialize Reddit instance
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        password= password,
    )
    
    # Create filename
    filename = f"{subreddit_name}_{keyword}.csv"
    
    try:
        # Get subreddit
        subreddit = reddit.subreddit(subreddit_name)
        
        # Search for posts with keyword (top posts from past year)
        posts = subreddit.search(keyword, limit=limit, sort='top', time_filter='year')
        
        # Open CSV file for writing
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['title', 'post_text', 'score', 'url', 'created_utc', 'author']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Counter for fetched posts
            count = 0
            
            # Iterate through posts
            for post in posts:
                try:
                    # Skip if not a text post (selftext should not be empty)
                    if not post.selftext or post.selftext.strip() == "":
                        continue
                    
                    # Get post text
                    post_text = post.selftext
                    
                    # Write row to CSV
                    writer.writerow({
                        'title': post.title,
                        'post_text': post_text,
                        'score': post.score,
                        'url': post.url,
                        'created_utc': post.created_utc,
                        'author': str(post.author) if post.author else '[deleted]'
                    })
                    
                    count += 1
                    
                    # Rate limiting to avoid hitting API limits
                    if count % 10 == 0:
                        time.sleep(1)
                        print(f"Fetched {count} posts...")
                        
                except Exception as e:
                    print(f"Error processing post: {e}")
                    continue
        
        print(f"Successfully saved {count} posts to {filename}")
        
    except Exception as e:
        print(f"Error: {e}")
        


def classify_post_advice_seeking(title: str, post_text: str, client) -> str:
    """
    Classifies a single post as advice-seeking or not using OpenAI API.
    
    Args:
        title (str): Post title
        post_text (str): Post content
        client: OpenAI client instance
        
    Returns:
        str: 'advice_seeking' or 'not_advice_seeking'
    """
    
    prompt = f"""
    Analyze the following Reddit post and determine if the author is seeking advice or help.

    Post Title: "{title}"
    Post Content: "{post_text}"

    A post is considered "advice-seeking" if the author is:
    - Asking for recommendations, suggestions, or guidance
    - Seeking help with a problem or decision
    - Requesting opinions on what to do in a situation
    - Looking for best practices or "how-to" information
    - Asking "should I..." or "what would you do..." type questions

    A post is "not advice-seeking" if it's:
    - Sharing information, news, or tutorials
    - Making announcements or statements
    - Discussing general topics without seeking input
    - Showing off projects or achievements
    - Just starting conversations without needing advice

    Respond with exactly one word: either "advice_seeking" or "not_advice_seeking"
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a classifier that determines if Reddit posts are seeking advice. Respond with exactly one word: 'advice_seeking' or 'not_advice_seeking'."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0
        )
        
        classification = response.choices[0].message.content.strip().lower()
        
        # Ensure we only return valid classifications
        if classification in ['advice_seeking', 'not_advice_seeking']:
            return classification
        else:
            # Default fallback if response is unexpected
            return 'not_advice_seeking'
            
    except Exception as e:
        print(f"Error classifying post: {e}")
        return 'error'

def classify_reddit_posts_csv(csv_file_path: str, api_key: str, 
                             output_file_path = None, 
                             batch_size: int = 10) -> None:
    """
    Reads a CSV file of Reddit posts, classifies them using OpenAI API, 
    and saves the results with an additional classification column.
    
    Args:
        csv_file_path (str): Path to the input CSV file
        api_key (str): OpenAI API key
        output_file_path (str): Path for output file (optional)
        batch_size (int): Number of posts to process before saving progress
    """
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} posts from {csv_file_path}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Create output filename if not provided
    if output_file_path is None:
        base_name = os.path.splitext(csv_file_path)[0]
        output_file_path = f"{base_name}_classified.csv"
    
    # Add classification column if it doesn't exist
    if 'advice_seeking' not in df.columns:
        df['advice_seeking'] = ''
    
    # Process posts
    total_posts = len(df)
    processed = 0
    
    for index, row in df.iterrows():
        # Skip if already classified
        if pd.notna(row['advice_seeking']) and row['advice_seeking'] != '':
            processed += 1
            continue
        
        print(f"Processing post {processed + 1}/{total_posts}: {row['title'][:50]}...")
        
        # Classify the post
        classification = classify_post_advice_seeking(
            title=row['title'],
            post_text=row['post_text'],
            client=client
        )
        
        # Update the dataframe
        df.at[index, 'advice_seeking'] = classification
        processed += 1
        
        # Save progress every batch_size posts
        if processed % batch_size == 0:
            df.to_csv(output_file_path, index=False)
            print(f"Saved progress: {processed}/{total_posts} posts classified")
        
        # Rate limiting to avoid hitting API limits
        time.sleep(1)  # 1 second delay between requests
    
    # Final save
    df.to_csv(output_file_path, index=False)
    
    # Print summary
    advice_count = len(df[df['advice_seeking'] == 'advice_seeking'])
    not_advice_count = len(df[df['advice_seeking'] == 'not_advice_seeking'])
    error_count = len(df[df['advice_seeking'] == 'error'])
    
    print(f"\nClassification Complete!")
    print(f"Total posts: {total_posts}")
    print(f"Advice-seeking posts: {advice_count}")
    print(f"Not advice-seeking posts: {not_advice_count}")
    print(f"Errors: {error_count}")
    print(f"Results saved to: {output_file_path}")

# Example usage


# Example usage
if __name__ == "__main__":
    # You need to replace these with your actual Reddit API credentials
    CLIENT_ID = "OLU35HaHuePcprWQ0SWMkQ"
    CLIENT_SECRET = "roce-zTUAPtF_utdExfDogewC3vHSg"
    USER_AGENT = "context_eval_scraper by u/context_evals"
    PASSWORD = "kmb-jpk!JNF2bud1trv"
    SUBREDDIT = "personalfinance"
    KEYWORD = "small investment"
    
    # Example function call
    fetch_reddit_posts_to_csv(
        subreddit_name=SUBREDDIT,
        keyword=KEYWORD,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
        password = PASSWORD,
        limit=10
    )

    # Set your OpenAI API key
    #API_KEY = "your_openai_api_key_here"  # Replace with your actual API key
    
    # Or get from environment variable (recommended)
    API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Example usage
    csv_file = f"{SUBREDDIT}_{KEYWORD}.csv"  # Your CSV file from the Reddit scraper
    
    classify_reddit_posts_csv(
        csv_file_path=csv_file,
        api_key=API_KEY,
        batch_size=5  # Save progress every 5 posts
    )