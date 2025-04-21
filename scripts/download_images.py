from icrawler.builtin import GoogleImageCrawler
import os

# Türler ve arama sorguları:
categories = {
    'Orkide':    ['orchid plant leaf', 'orchid leaf closeup'],
    'Monstera':  ['monstera plant leaf', 'monstera deliciosa leaf'],
    'Aloe_vera': ['aloe vera plant leaf', 'aloe vera closeup'],
    'Kaktus':    ['cactus plant leaf', 'cactus closeup']
}

root = os.getcwd()  # Proje kökü

for species, query_list in categories.items():
    out_dir = os.path.join(root, 'data', 'raw', species)
    os.makedirs(out_dir, exist_ok=True)

    for query in query_list:
        print(f"\n🔽 Downloading for {species} → Query: {query}")
        crawler = GoogleImageCrawler(storage={'root_dir': out_dir}, log_level='INFO')
        crawler.crawl(keyword=query, max_num=80, overwrite=False)
