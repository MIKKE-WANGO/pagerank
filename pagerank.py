import os
import random
import re
import sys
import random
DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    print(corpus)
     
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

    
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
       print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    # Initialize the probability distribution
    prob_dist = dict()
    pages = corpus.keys()
    #print(pages)

    # If the page has no outgoing links, return a probability distribution that chooses randomly among all pages with equal probability.
    if len(corpus[page]) == 0:
        equal_prob = 1/len(pages)
        for page in pages:
            prob_dist[page] = equal_prob
    
    #page has outgoing links
    else:
        remaining_prob = 1 - damping_factor
       
        for link in corpus[page]:
            prob_dist[link] = damping_factor / len(corpus[page]) + remaining_prob/len(corpus)
        
        for link in pages:
            if link not in prob_dist:
                prob_dist[link] = remaining_prob/len(corpus)

    
    #print(prob_dist)
    return prob_dist

    #raise NotImplementedError


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    #get start page
    pages = corpus.keys()
    start_page = random.choice(list(pages))
    print('sample function')
    print(start_page)

    #dict to keep track of occurences of each page
    pages_occurence = dict()
    for page in pages:
        pages_occurence[page] = 0
    print(pages_occurence)

    
    i = 0
    chosen_page = start_page
    pages_occurence[chosen_page] = pages_occurence[chosen_page] + 1
    
    while i < n-1:
        sample = transition_model(corpus,chosen_page,damping_factor)
        sample_pages = list(sample.keys())
        weights = list(sample.values())
        # Choose a page randomly based on weights
        chosen_page = random.choices(sample_pages, weights=weights, k=1)[0] 
        pages_occurence[chosen_page] = pages_occurence[chosen_page] + 1
        i += 1
        
    
    pagerank = dict()
    for page in pages_occurence:
        pagerank[page] = pages_occurence[page]/n
    
    print(pages_occurence)
    print(pagerank)
    print(i)
   
    return pagerank



        

    
    #raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    init_rank = 1 / num_pages
    random_choice_prob = (1 - damping_factor) / len(corpus)
    iterations = 0

    # Initial page_rank gives every page a rank of 1/(num pages in corpus)
    page_ranks = {page_name: init_rank for page_name in corpus}
    new_ranks = {page_name: 0 for page_name in corpus}
    max_rank_change = init_rank

    while max_rank_change > 0.001:
        
        max_rank_change = 0
        for page in page_ranks:
            sumation = 0 
            for page1 in page_ranks:
                if page in corpus[page1]:
                    prob = page_ranks[page1]/len(corpus[page1])
                    sumation += prob
                elif len(corpus[page1]) == 0:
                    prob = page_ranks[page1]/num_pages
                    sumation += prob
            
            new_page_rank = random_choice_prob + (damping_factor * sumation)
            new_ranks[page] = new_page_rank

        # Normalise the new page ranks:
        norm_factor = sum(new_ranks.values())
        new_ranks = {page: (rank / norm_factor) for page, rank in new_ranks.items()}

        # Find max change in page rank:
        for page_name in corpus:
            rank_change = abs(page_ranks[page_name] - new_ranks[page_name])
            if rank_change > max_rank_change:
                max_rank_change = rank_change

        # Update page ranks to the new ranks:
        #use copy to ensure it creates a new dictionary instead of referencing 
        page_ranks = new_ranks.copy()
        iterations += 1
    print(iterations, 'iterations to converge')
    print('Sum of iteration page ranks: ', round(sum(page_ranks.values()), 4))

    return page_ranks

        






    #raise NotImplementedError


if __name__ == "__main__":
    main()
