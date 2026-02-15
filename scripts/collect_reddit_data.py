"""
Step 1: Collect parenting posts + top comments from Reddit.

Uses PRAW to fetch top posts from dad/parenting subreddits.
Saves raw data as JSONL for downstream processing.

Fixes over v1:
- Fixed time_filter bug (was hitting same endpoint 10x)
- Uses time_filter="all" to get best posts across all time
- Applies min_score filter (was defined but never used)
- Deduplicates by permalink
- Filters [deleted], [removed], bot comments
- Correct output path (data/ directory)
- Proper error handling and env var validation
"""

import praw
import os
import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def validate_env_vars():
    """Check that all required Reddit API credentials are set."""
    required = [
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "REDDIT_USER_AGENT",
    ]
    missing = [var for var in required if not os.getenv(var)]
    if missing:
        print(f"Error: Missing environment variables: {', '.join(missing)}")
        print("Create a .env file with your Reddit API credentials.")
        print("See: https://www.reddit.com/prefs/apps to create an app.")
        raise SystemExit(1)


def create_reddit_client():
    """Create and return an authenticated Reddit client."""
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT"),
        username=os.getenv("REDDIT_USERNAME"),
        password=os.getenv("REDDIT_PASSWORD"),
    )


# Patterns that indicate bot/auto-generated/removed content
BOT_PATTERNS = [
    "i am a bot",
    "i'm a bot",
    "this action was performed automatically",
    "automoderator",
    "remindmebot",
    "bot action",
    "this is an automated",
]

REMOVED_PATTERNS = [
    "[deleted]",
    "[removed]",
    "[unavailable]",
]


def is_low_quality_comment(text):
    """Check if a comment is from a bot, removed, or too short."""
    lower = text.lower().strip()

    # Check for removed/deleted
    if lower in [p.lower() for p in REMOVED_PATTERNS]:
        return True

    # Check for bot patterns
    for pattern in BOT_PATTERNS:
        if pattern in lower:
            return True

    return False


def get_top_posts(reddit, subreddit_name, max_posts=250, min_score=10,
                  min_comment_length=100, time_filters=None):
    """
    Fetch top posts with quality comments from a subreddit.

    Uses multiple time_filter values to maximize coverage:
    - "all" gets the best posts ever
    - "year" gets recent popular posts
    - "month" catches trending content

    Args:
        reddit: PRAW Reddit client
        subreddit_name: Name of the subreddit
        max_posts: Maximum posts to collect
        min_score: Minimum post score (upvotes)
        min_comment_length: Minimum comment length in characters
        time_filters: List of time filters to use
    """
    if time_filters is None:
        time_filters = ["all", "year", "month"]

    seen_permalinks = set()
    valid_posts = []
    subreddit = reddit.subreddit(subreddit_name)

    for time_filter in time_filters:
        if len(valid_posts) >= max_posts:
            break

        try:
            print(f"  Fetching r/{subreddit_name} top/{time_filter}...")
            top_posts = subreddit.top(time_filter=time_filter, limit=500)

            for post in top_posts:
                if len(valid_posts) >= max_posts:
                    break

                # Deduplicate
                if post.permalink in seen_permalinks:
                    continue
                seen_permalinks.add(post.permalink)

                # Skip low-score posts
                if post.score < min_score:
                    continue

                # Skip posts without text content
                if not post.selftext or len(post.selftext.strip()) < 20:
                    continue

                # Get comments
                try:
                    post.comments.replace_more(limit=0)
                    if not post.comments:
                        continue

                    # Try top 3 comments to find a quality one
                    best_comment = None
                    for comment in post.comments[:3]:
                        body = comment.body.strip()
                        if (len(body) >= min_comment_length
                                and not is_low_quality_comment(body)
                                and comment.score >= 2):
                            best_comment = comment
                            break

                    if best_comment is None:
                        continue

                except Exception:
                    continue

                valid_posts.append({
                    "subreddit": subreddit_name,
                    "title": post.title,
                    "selftext": post.selftext.strip(),
                    "post_score": post.score,
                    "comment": best_comment.body.strip(),
                    "comment_score": best_comment.score,
                    "permalink": post.permalink,
                })

        except Exception as e:
            print(f"  Warning: Error on r/{subreddit_name} ({time_filter}): {e}")
            time.sleep(2)

    return valid_posts


def main():
    parser = argparse.ArgumentParser(description="Collect parenting posts from Reddit")
    parser.add_argument("--output", default="data/reddit_dataset.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--max-per-sub", type=int, default=250,
                        help="Max posts per subreddit")
    parser.add_argument("--min-score", type=int, default=10,
                        help="Minimum post score")
    parser.add_argument("--min-comment-length", type=int, default=100,
                        help="Minimum comment length in characters")
    args = parser.parse_args()

    # Validate credentials
    validate_env_vars()

    # Connect to Reddit
    reddit = create_reddit_client()
    print(f"Connected to Reddit as: {reddit.user.me() or 'read-only'}\n")

    # Subreddits to collect from
    # v2 originals (7 subs)
    # v3.1 expansion: +6 subs for broader coverage
    subreddits = [
        # --- v2 originals ---
        "NewDads",
        "Daddit",
        "BabyBumps",
        "Parenting",
        "predaddit",          # Dads-to-be (pregnancy stage)
        "beyondthebump",      # Post-birth parenting
        "newparents",         # New parent struggles
        # --- v3.1 additions ---
        "breakingdad",        # Raw, unfiltered dad content
        "SAHP",               # Stay-at-home parents — isolation, identity
        "SingleDads",         # Single fatherhood challenges
        "AskParents",         # Q&A format — ideal for instruction pairs
        "DadForAMinute",      # Wholesome support — perfect DadAI tone
        "AttachmentParenting", # Emotional, bonded parenting wisdom
    ]

    all_data = []
    global_seen = set()

    print("Starting Reddit collection...\n")
    for subreddit_name in tqdm(subreddits, desc="Subreddits"):
        posts = get_top_posts(
            reddit,
            subreddit_name,
            max_posts=args.max_per_sub,
            min_score=args.min_score,
            min_comment_length=args.min_comment_length,
        )

        # Global deduplication across subreddits
        new_posts = []
        for p in posts:
            if p["permalink"] not in global_seen:
                global_seen.add(p["permalink"])
                new_posts.append(p)

        print(f"  r/{subreddit_name}: {len(new_posts)} posts collected\n")
        all_data.extend(new_posts)

    # Save to JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nDone! {len(all_data)} posts saved to {output_path}")
    print(f"Subreddits: {', '.join(subreddits)}")
    print(f"Filters: min_score={args.min_score}, min_comment_length={args.min_comment_length}")


if __name__ == "__main__":
    main()
