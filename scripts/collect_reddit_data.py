import praw
import os
import json
from datetime import datetime
from tqdm import tqdm
import time
from dotenv import load_dotenv
from pathlib import Path

# 📂 Charge les variables d'environnement
load_dotenv()

# 🔐 Connexion Reddit via PRAW
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD")
)

# 🔧 Paramètres
subreddits = ["NewDads", "Daddit", "BabyBumps", "Parenting"]
max_posts_per_sub = 250
min_score = 10
output_file = Path(__file__).parent.parent / "data" / "reddit_dataset.jsonl"
# output_file = "reddit_dataset.jsonl"
max_years_back = 10

def get_top_posts(subreddit_name, limit_per_year, max_years=10):
    """🔁 Récupère les top posts valides d’un subreddit, année par année"""
    valid_posts = []
    current_year = datetime.now().year
    attempts = 0

    for year in range(current_year, current_year - max_years, -1):
        if len(valid_posts) >= limit_per_year:
            break

        try:
            print(f"🔍 {subreddit_name} - Année {year}...")
            subreddit = reddit.subreddit(subreddit_name)
            top_posts = subreddit.top(time_filter="year", limit=100)

            for post in top_posts:
                # Skip les posts d’autres années
                post_year = datetime.fromtimestamp(post.created_utc).year
                if post_year != year:
                    continue

                # Skip si déjà assez de posts
                if len(valid_posts) >= limit_per_year:
                    break

                # Vérifie la présence de commentaires
                try:
                    post.comments.replace_more(limit=0)
                    top_comment = post.comments[0] if post.comments else None
                    if top_comment is None or len(top_comment.body.strip()) == 0:
                        continue
                except Exception:
                    continue

                # Enregistre le post
                valid_posts.append({
                    "subreddit": subreddit_name,
                    "year": year,
                    "title": post.title,
                    "selftext": post.selftext,
                    "score": post.score,
                    "comment": top_comment.body.strip(),
                    "permalink": post.permalink
                })

        except Exception as e:
            print(f"⚠️  Erreur sur {subreddit_name} ({year}) : {e}")
            time.sleep(2)  # petite pause en cas d'erreur Reddit

    return valid_posts

# 🧠 Lancement
all_data = []
print("📡 Lancement de la collecte Reddit...\n")

for subreddit in tqdm(subreddits, desc="🔁 Subreddits"):
    posts = get_top_posts(subreddit, limit_per_year=max_posts_per_sub, max_years=max_years_back)
    print(f"✅ {subreddit} : {len(posts)} posts collectés\n")
    all_data.extend(posts)

# 💾 Sauvegarde en JSONL
with open(output_file, "w") as f:
    for item in all_data:
        f.write(json.dumps(item) + "\n")

print(f"\n🎉 Terminé ! {len(all_data)} posts sauvegardés dans {output_file}")