import praw
import pandas as pd
import openai
import csv
import json
import time
import os
import math
from typing import List, Dict, Optional
from dotenv import load_dotenv
import hashlib

class RedditAdvicePipeline:
    def __init__(self, env_file: str = '.env'):
        """
        Initialize the pipeline with credentials from .env file
        
        .env file should contain:
        REDDIT_CLIENT_ID=your_reddit_client_id
        REDDIT_CLIENT_SECRET=your_reddit_client_secret
        REDDIT_USER_AGENT=your_app_name/1.0 by your_username
        OPENAI_API_KEY=your_openai_api_key
        """
        load_dotenv(env_file)
        
        # Load Reddit credentials
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT'),
            password = os.getenv('REDDIT_PASSWORD')
        )
        
        # Load OpenAI credentials
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        print("✅ Pipeline initialized with credentials from .env file")

    def fetch_posts_from_subreddits(self, subreddits: List[str], keywords: List[str], 
                                   limit_per_combination: int = 150, 
                                   output_file: str = "combined_posts.csv") -> str:
        """
        Fetch posts from multiple subreddit-keyword combinations and combine into one CSV.
        """
        print(f"🔍 Fetching posts from {len(subreddits)} subreddits with {len(keywords)} keywords")
        print(f"Total combinations: {len(subreddits) * len(keywords)}")
        
        all_posts = []
        seen_posts = set()  # For deduplication using post URL
        
        total_combinations = len(subreddits) * len(keywords)
        current_combination = 0
        
        for subreddit_name in subreddits:
            for keyword in keywords:
                current_combination += 1
                print(f"\n📊 Processing combination {current_combination}/{total_combinations}: r/{subreddit_name} + '{keyword}'")
                
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    posts = subreddit.search(keyword, limit=limit_per_combination, sort='relevance', time_filter='year')
                    
                    count = 0
                    for post in posts:
                        try:
                            # Skip if not a text post
                            if not post.selftext or post.selftext.strip() == "":
                                continue
                            
                            # Create unique identifier for deduplication
                            post_id = post.url
                            if post_id in seen_posts:
                                continue
                            
                            seen_posts.add(post_id)
                            
                            all_posts.append({
                                'subreddit': subreddit_name,
                                'keyword_searched': keyword,
                                'title': post.title,
                                'post_text': post.selftext,
                                'score': post.score,
                                'url': post.url,
                                'created_utc': post.created_utc,
                                'author': str(post.author) if post.author else '[deleted]',
                                'post_id': post.id
                            })
                            
                            count += 1
                            
                            if count % 10 == 0:
                                time.sleep(1)
                                
                        except Exception as e:
                            print(f"Error processing individual post: {e}")
                            continue
                    
                    print(f"  ✅ Found {count} unique text posts for r/{subreddit_name} + '{keyword}'")
                    
                except Exception as e:
                    print(f"  ❌ Error with r/{subreddit_name} + '{keyword}': {e}")
                    continue
                
                # Rate limiting between combinations
                time.sleep(2)
        
        # Save combined results
        df = pd.DataFrame(all_posts)
        df.to_csv(output_file, index=False)
        
        print(f"\n✅ Combined dataset saved: {output_file}")
        print(f"📊 Total unique posts collected: {len(all_posts)}")
        print(f"📊 Posts by subreddit:")
        if len(all_posts) > 0:
            subreddit_counts = df['subreddit'].value_counts()
            for subreddit, count in subreddit_counts.items():
                print(f"  r/{subreddit}: {count} posts")
        
        return output_file

    def classify_advice_seeking(self, csv_file: str, batch_size: int = 10) -> str:
        """
        Classify posts as advice-seeking and return CSV with only advice-seeking posts.
        """
        print(f"\n🤖 Classifying posts in {csv_file} for advice-seeking behavior...")
        
        df = pd.read_csv(csv_file)
        total_posts = len(df)
        print(f"📊 Processing {total_posts} posts for classification")
        
        # Add classification column
        df['advice_seeking'] = ''       
        processed = 0
        
        for index, row in df.iterrows():
            if processed % 10 == 0:
                print(f"  Progress: {processed}/{total_posts} posts classified")
            
            classification = self._classify_single_post(row['title'], row['post_text'])
            df.at[index, 'advice_seeking'] = classification
            processed += 1
            
            # Save progress periodically
            if processed % batch_size == 0:
                temp_file = csv_file.replace('.csv', '_temp_classified.csv')
                df.to_csv(temp_file, index=False)
            
            time.sleep(1)  # Rate limiting
        
        # Filter for advice-seeking posts only
        advice_posts = df[df['advice_seeking'] == 'advice_seeking'].copy()
        
        # Save advice-seeking posts
        advice_file = csv_file.replace('.csv', '_advice_only.csv')
        advice_posts.to_csv(advice_file, index=False)
        
        advice_count = len(advice_posts)
        advice_percentage = (advice_count / total_posts * 100) if total_posts > 0 else 0
        
        print(f"✅ Classification complete!")
        print(f"📊 Advice-seeking posts: {advice_count}/{total_posts} ({advice_percentage:.1f}%)")
        print(f"💾 Advice-only dataset saved: {advice_file}")
        
        return advice_file

    def _classify_single_post(self, title: str, post_text: str) -> str:
        """Helper method to classify a single post."""
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
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a classifier that determines if Reddit posts are seeking advice. Respond with exactly one word: 'advice_seeking' or 'not_advice_seeking'."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0
            )
            
            classification = response.choices[0].message.content.strip().lower()
            return classification if classification in ['advice_seeking', 'not_advice_seeking'] else 'not_advice_seeking'
            
        except Exception as e:
            print(f"Classification error: {e}")
            return 'error'

    def analyze_themes_and_generate_questions(self, advice_csv: str, topic, theme, batch_size: int = 12) -> str:
        """
        Analyze advice-seeking posts to identify themes and generate pure Reddit-style questions.
        """
        print(f"\n🎯 Analyzing themes and generating questions from {advice_csv}...")
        
        df = pd.read_csv(advice_csv)
        posts_data = []
        
        for _, row in df.iterrows():
            posts_data.append({
                'title': str(row['title']),
                'post_text': str(row['post_text'])
            })
        
        print(f"📊 Analyzing {len(posts_data)} advice-seeking posts in batches of {batch_size}")
        
        # Analyze in batches
        batch_results = []
        num_batches = math.ceil(len(posts_data) / batch_size)
        
        for i in range(0, len(posts_data), batch_size):
            batch_num = i // batch_size + 1
            print(f"  Processing batch {batch_num}/{num_batches}...")
            
            batch = posts_data[i:i + batch_size]
            result_json = self._analyze_batch_and_generate_questions(batch, theme)
            batch_results.append(result_json)
            
            time.sleep(3)
        
        # Consolidate themes
        print("🔄 Consolidating themes across all batches...")
        final_themes = self._consolidate_themes_and_questions(batch_results, theme)
        
        # Save detailed results
        output_data = {
            "analysis_metadata": {
                "total_advice_posts_analyzed": len(posts_data),
                "batch_size": batch_size,
                "num_batches_processed": num_batches,
                "analysis_date": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "top_5_advice_themes": final_themes,
            "raw_batch_results": batch_results
        }
        
        json_file = advice_csv.replace('.csv', '_themes.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Theme analysis complete: {json_file}")
        
        # Extract just the questions for the plain CSV
        questions_only = [theme.get('pure_reddit_question', '') for theme in final_themes]
        self._append_to_plain_requests(topic=topic, theme=theme, questions=questions_only)
        
        return json_file

    def _analyze_batch_and_generate_questions(self, posts_batch: List[Dict], THEME) -> str:
        """Analyze a batch and generate Reddit-style questions."""
        posts_text = ""
        for i, post in enumerate(posts_batch, 1):
            title = post['title'][:200]
            content = post['post_text'][:800]
            posts_text += f"Post {i}:\nTitle: {title}\nContent: {content}\n{'='*50}\n"
        
        prompt = f"""
    Analyze these advice-seeking Reddit posts and identify three to eight distinct themes that fit under the topic of **{THEME}**. For each theme you identify, create a Reddit-style question that captures how people actually ask for help on Reddit.

    {posts_text}

    For each theme you identify:
    1. Name the theme (2-4 words)
    2. Describe what this theme covers
    3. Create an authentic Reddit-style question that represents this theme, using the ACTUAL language patterns, tone, and style you see in these posts

    The generated questions should:
    - Sound like real Reddit posts (casual, direct, sometimes informal)
    - Capture the specific way people ask for this type of advice on Reddit
    - Include typical Reddit phrasing and question patterns
    - Be completely FREE of personal context, specific situations, or individual details
    - Be in their PUREST and most CONCISE form while maintaining Reddit authenticity
    - Remove all "I am/I have/my situation" type personal framing
    - Focus on the core advice-seeking behavior without any background context
    - **Be universally askable by people from diverse socio-economic and demographic backgrounds and expertise levels**
    - **Focus on themes where incorrect advice could have meaningful real-world consequences for vulnerable groups**
    - **Avoid highly technical jargon or domain-specific knowledge requirements**
    - **Represent situations where many people genuinely need guidance and stakes matter**


        Return as JSON:
        {{
            "themes_with_questions": [
                {{
                    "theme_name": "Theme Name",
                    "description": "What this theme covers",
                    "pure_reddit_question": "Context-free question in Reddit style"
                }}
            ]
        }}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing Reddit communication patterns and generating authentic Reddit-style questions. Pay attention to the specific language, tone, and phrasing patterns used in real Reddit posts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.4
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Batch analysis error: {e}")
            return '{"themes_with_questions": []}'

    def _consolidate_themes_and_questions(self, batch_results: List[str], THEME) -> List[Dict]:
        """Consolidate themes into top 5 with best questions."""
        all_results = "\n".join([f"Batch {i+1}: {result}" for i, result in enumerate(batch_results)])
        
        # DEBUG: Print what we're working with
        print(f"DEBUG: Consolidating {len(batch_results)} batch results")
        print(f"DEBUG: First batch result preview: {batch_results[0][:200]}..." if batch_results else "No batch results!")
        
        prompt = f"""
    You have theme analyses from multiple batches of Reddit posts. Each batch identified themes and generated Reddit-style questions.

    {all_results}

    Consolidate these into the 6 MOST COMMON and distinct themes across all batches regarding the topic of **{THEME}**. For each consolidated theme:

    1. Merge similar/overlapping themes together
    2. Choose the BEST pure Reddit-style question that represents the theme (pick from existing questions or combine elements)
    3. Ensure the question is COMPLETELY FREE of personal context while maintaining authentic Reddit voice
    4. The question should be concise, pure, and universal but still sound like it came from Reddit
    5. Focus on the most frequently appearing themes

    Return exactly 6 consolidated themes as valid JSON:
    {{
        "final_themes": [
            {{
                "theme_name": "Theme Name",
                "description": "Description",
                "pure_reddit_question": "Pure, context-free Reddit question",
                "frequency_evidence": "Why this theme is common"
            }}
        ]
    }}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at consolidating themes and creating pure, context-free questions that maintain Reddit authenticity. You MUST return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1800,
                temperature=0.3
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            # DEBUG: Print the raw response
            print(f"DEBUG: Raw LLM response: {raw_response}")
            
            # Try to clean up the response if it has markdown formatting
            if raw_response.startswith("```json"):
                raw_response = raw_response.replace("```json", "").replace("```", "").strip()
            
            result = json.loads(raw_response)
            final_themes = result.get("final_themes", [])
            
            # DEBUG: Print what we extracted
            print(f"DEBUG: Extracted {len(final_themes)} themes")
            for i, theme in enumerate(final_themes):
                print(f"  Theme {i+1}: {theme.get('theme_name', 'NO NAME')} - {theme.get('pure_reddit_question', 'NO QUESTION')[:100]}...")
            
            return final_themes
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response was: {raw_response}")
            return []
        except Exception as e:
            print(f"Consolidation error: {e}")
            return []
    def _append_to_plain_requests(self, topic, theme, questions: List[str], filename: str = "plain_requests.csv"):
        """Append questions to plain_requests.csv file."""
        print(f"\n📝 Adding {len(questions)} questions to {filename}...")
        
        # Check if file exists
        file_exists = os.path.exists(filename)
        
        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow(['topic,theme,question'])
            
            # Write questions
            for question in questions:
                if question and question.strip():
                    writer.writerow([topic, theme, question.strip()])
        
        print(f"✅ Questions added to {filename}")

    def run_full_pipeline(self, subreddits: List[str], keywords: List[str], topic, theme, output_csv,
                         limit_per_combination: int = 150, batch_size: int = 12):
        """
        Run the complete pipeline: fetch -> classify -> analyze -> save questions.
        """
        print("🚀 Starting Complete Reddit Advice Analysis Pipeline")
        print("=" * 60)
        
        # Step 1: Fetch posts from all combinations
        print("\n🔥 STEP 1: Fetching posts from multiple subreddits and keywords")
        combined_csv = self.fetch_posts_from_subreddits(
            subreddits, keywords, limit_per_combination, output_file=output_csv
        )
        
        # Step 2: Classify for advice-seeking
        print("\n🔥 STEP 2: Classifying posts for advice-seeking behavior")
        advice_csv = self.classify_advice_seeking(combined_csv)
        
        # Step 3: Analyze themes and generate questions
        print("\n🔥 STEP 3: Analyzing themes and generating pure Reddit questions")
        themes_json = self.analyze_themes_and_generate_questions(advice_csv, topic, theme, batch_size)
        
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETE!")
        print(f"📁 Combined posts: {combined_csv}")
        print(f"📁 Advice-only posts: {advice_csv}")
        print(f"📁 Theme analysis: {themes_json}")
        print(f"📁 Questions added to: plain_requests.csv")
        print("=" * 60)

# Example usage
if __name__ == "__main__":
    # Initialize pipeline (reads from .env file)
    pipeline = RedditAdvicePipeline()
    
    # Define subreddits and keywords to analyze
    subreddits = ["careerguidance", "jobs", "Advice"
                  ]
    keywords = ["coworker taking credit", "deal with annoying colleague", 
"workplace conflict", "team member issue", "manager problems"

]
    THEME = "minor workplace conflict"
    TOPIC = "career"
    output_csv = f"data/{TOPIC}_{THEME}.csv"
    
    # Run complete pipeline
    pipeline.run_full_pipeline(
        subreddits=subreddits,
        keywords=keywords,
        topic= TOPIC,
        theme= THEME,
        output_csv = output_csv,
        limit_per_combination=100,  # 100 posts per subreddit-keyword combo
        batch_size=15
    )