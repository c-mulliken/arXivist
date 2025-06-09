from transformers import AutoTokenizer
from adapters import AutoAdapterModel

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from arxiv_pipeline import get_results

tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
model.eval()

def get_embeddings(papers):
    model.load_adapter("allenai/specter2", source="hf", load_as='specter2', set_active=True)
    tbatch = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in papers]
    inputs = tokenizer(tbatch, padding=True, truncation=True,
                       return_tensors="pt", return_token_type_ids=False, max_length=512)
    output = model(**inputs)
    return output.last_hidden_state[:, 0, :].detach().numpy()

def get_query(query):
    model.load_adapter("allenai/specter2_adhoc_query", source="hf", load_as="specter2_adhoc_query", set_active=True)
    inputs = tokenizer(query, padding=True, truncation=True,
                       return_tensors="pt", return_token_type_ids=False, max_length=512)
    output = model(**inputs)
    return output.last_hidden_state[:, 0, :].detach().numpy()

if __name__ == "__main__":
    ml_papers = get_results(query='cat:cs.LG', max_results=100)
    query = ["Transformers"]
    query_emb = get_query(query)
    print(query_emb.shape)
    papers_emb = get_embeddings(ml_papers)
    print(papers_emb.shape)
    similarities = cosine_similarity(query_emb, papers_emb)
    print(similarities.shape)
    # get 5 most similar papers
    most_similar_indices = similarities[0].argsort()[-5:][::-1]
    most_similar_papers = [ml_papers[i] for i in most_similar_indices]
    for i, paper in enumerate(most_similar_papers):
        print(f"{i+1}. {paper['title']}")