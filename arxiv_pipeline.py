import arxiv

def get_results(query="cat:cs.LG", max_results=40):
    search = arxiv.Search(
        query="cat:cs.LG",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    result_dict = [
        {'title': r.title, 'abstract': r.summary} for r in search.results()
    ]
    return result_dict