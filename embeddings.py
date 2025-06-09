from transformers import AutoTokenizer
from adapters import AutoAdapterModel

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from arxiv_pipeline import get_results

tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
adapter_name = model.load_adapter("allenai/specter2", source="hf", load_as='specter2', set_active=True)
model.eval()

ml_papers = get_results(query='cat:cs.LG', max_results=40)
phys_papers = get_results(query='cat:astrop-ph.CO', max_results=40)

def get_embeddings(papers):
    tbatch = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in papers]
    inputs = tokenizer(tbatch, padding=True, truncation=True,
                       return_tensors="pt", return_token_type_ids=False, max_length=512)
    output = model(**inputs)
    return output.last_hidden_state[:, 0, :].detach().numpy()

ml_emb = get_embeddings(ml_papers)
phys_emb = get_embeddings(phys_papers)

import numpy as np
print("ML embeddings shape:", ml_emb.shape)
print("Physics embeddings shape:", phys_emb.shape)

all_emb = np.vstack([ml_emb, phys_emb])
labels = np.array([0]*len(ml_emb) + [1]*len(phys_emb))

pca = PCA(n_components=2)
reduced = pca.fit_transform(all_emb)
print("Reduced shape:", reduced.shape)
print("Labels:", labels)

plt.scatter(reduced[labels==0, 0], reduced[labels==0, 1], color='blue', label='ML', marker='o', alpha=0.7)
plt.scatter(reduced[labels==1, 0], reduced[labels==1, 1], color='red', label='Physics', marker='^', alpha=0.7)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA of ML vs Physics Paper Embeddings')
plt.legend()
plt.tight_layout()
plt.show()