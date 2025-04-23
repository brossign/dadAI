import os
import dotenv # Pour charger les variables d'environnement du fichier .env
import praw # API Reddit

# Charge les variables depuis .env
dotenv.load_dotenv()

# Récupère les variables d'environnement
client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = os.getenv("REDDIT_USER_AGENT")

# Initialise la connexion à Reddit
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

# 🔎 TEST : accès à un subreddit public
subreddit = reddit.subreddit("Daddit")

# Affiche les 5 posts les plus populaires
print("\nTop 5 posts dans r/Daddit:\n")
for post in subreddit.hot(limit=5):
    print(f"📌 {post.title} (score: {post.score})")