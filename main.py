import argparse
from pathlib import Path

import smart_open
from gensim.corpora.wikicorpus import WikiCorpus, tokenize
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

class TaggedWikiCorpus:
    def __init__(self, wiki_text_path):
        self.wiki_text_path = wiki_text_path
        
    def __iter__(self):
        for line in smart_open.open(self.wiki_text_path, encoding='utf8'):
            title, words = line.split('\t')
            yield TaggedDocument(words=words.split(), tags=[title])

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Doc2Vec model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--vector_size", type=int, default=100, help="Dimensionality of the feature vectors.")
    return parser.parse_args()

def dowload_wiki_dump():
    import requests
    print("Downloading Wikipedia dump...")
    url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
    response = requests.get(url, stream=True)
    with open(DATA_DIR / "enwiki-latest-pages-articles.xml.bz2", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

def create_tagged_corpus(wiki_text_path):
    raw_wiki_file = DATA_DIR / "enwiki-latest-pages-articles.xml.bz2"
    if not raw_wiki_file.exists():
        dowload_wiki_dump()
    wiki = WikiCorpus(
        DATA_DIR / "enwiki-latest-pages-articles.xml.bz2",  # path to the file you downloaded above
        tokenizer_func=tokenize,  # simple regexp; plug in your own tokenizer here
        metadata=True,  # also return the article titles and ids when parsing
        dictionary={},  # don't start processing the data yet
    )
    with smart_open.open(DATA_DIR / "wiki.txt.gz", "w", encoding='utf8') as fout:
        for article_no, (content, (page_id, title)) in tqdm(enumerate(wiki.get_texts()), total=6_000_000):
            title = ' '.join(title.split())
            fout.write(f"{title}\t{' '.join(content)}\n")  # title_of_article [TAB] words of the article

def get_tagged_corpus():
    wiki_text_path = DATA_DIR / "wiki_text.txt"
    if not wiki_text_path.exists():
        create_tagged_corpus(wiki_text_path)
    return TaggedWikiCorpus(wiki_text_path)

def train_model(tagged_corpus, vector_size, epochs):
    model = Doc2Vec(vector_size=vector_size, min_count=2, epochs=epochs)
    model.build_vocab(tagged_corpus)
    model.train(tagged_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(MODEL_DIR / f"doc2vec_{vector_size}d.model")

def main():
    args = parse_args()
    print(f"Training Doc2Vec model with {args.epochs} epochs and vector size {args.vector_size}...")
    tagged_corpus = get_tagged_corpus()
    train_model(tagged_corpus, args.vector_size, args.epochs)
    print("Training complete! Model saved.")
    
if __name__ == "__main__":
    main()