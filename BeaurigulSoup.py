import requests
from bs4 import BeautifulSoup
import json

url = 'https://docs.arenadata.io/en/landing-adb/index.html'
headers = {
    'User-Agent': 'Mozilla/5.0'
}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

data = []

for point in soup.find_all("div", class_="product-roadmap-point"):
    label = point.find("div", class_="prp-label")
    version = label.get_text(strip=True) if label else ""
    date = point.get("data-date", "")
    ul = point.find("ul", class_="prp-release-notes")
    changes = []
    if ul:
        for li in ul.find_all("li"):
            changes.append(li.get_text(strip=True))
    link = ""
    link_wrap = point.find("div", class_="prp-description__doc-link-wrap")
    if link_wrap and link_wrap.find("a"):
        link = link_wrap.find("a")["href"]

    if changes:
        data.append({
            "version": version,
            "date": date,
            "changes": changes,
            "details_link": link
        })

# Сохраняем в JSONL
with open("release_notes.jsonl", "w", encoding="utf-8") as f:
    for entry in data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Сохранено {len(data)} релизов в release_notes.jsonl")
