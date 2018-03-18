

import json
import urllib.request


class LiveNews:
    def __init__(self, API_KEY):
        self.API_KEY = API_KEY

    def fetch(self, number_of_articles=100, sort_by="top", source="bbc-news"):
        data = []
        with urllib.request.urlopen("https://newsapi.org/v1/sources") as url:
            sources_data = json.loads(url.read().decode())
            all_sources_list = list(s['name'] for s in sources_data["sources"])
        # print("https://newsapi.org/v1/articles?source=" + source + "&sortBy=" + sort_by + "&apiKey=" + self.API_KEY)
        with urllib.request.urlopen(
            "https://newsapi.org/v1/articles?source=" + source + "&sortBy=" + sort_by + "&apiKey=" + self.API_KEY) as url:
            data = json.loads(url.read().decode())

        return list({"title":article["title"], "url":article["url"]} for article in data["articles"])[:number_of_articles]
