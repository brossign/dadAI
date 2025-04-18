import requests
from bs4 import BeautifulSoup
import json
import time

BASE_URL = "https://community.babycenter.com"
START_PAGE = "/groups/a6773784/december_2023_birth_club"

headers = {
    "User-Agent": "Mozilla/5.0"
}

posts = []

def scrape_topic_list(url):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Utilisation d'un sélecteur plus générique pour les liens de discussion
    links = soup.find_all("a", href=True)
    thread_links = [link for link in links if "/post/" in link['href']]
    print(f"Found {len(thread_links)} threads.")

    for link in thread_links[:5]:  # Limite à 5 pour les tests
        topic_url = BASE_URL + link['href']
        post = scrape_topic(topic_url)
        if post:
            posts.append(post)
        time.sleep(1)

def scrape_topic(url):
    print(f"Scraping: {url}")
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # Recherche du titre
    title_tag = soup.find("h1")
    if not title_tag:
        title_tag = soup.select_one(".discussion_topic_title")

    # Recherche du contenu
    content_tag = soup.select_one(".message_text")

    if not title_tag or not content_tag:
        print(f"Skipped (missing content): {url}")
        return None

    title = title_tag.get_text(strip=True)
    content = content_tag.get_text(strip=True)

    return {
        "title": title,
        "content": content,
        "url": url
    }

# Exécution du scraper
scrape_topic_list(BASE_URL + START_PAGE)

# Sauvegarde des résultats
output_path = "data/babycenter_sample.json"
with open(output_path, "w") as f:
    json.dump(posts, f, indent=2)

print(f"✅ Scraped {len(posts)} posts.")