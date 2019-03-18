import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

def sentimentToWordlist(rawReview, removeStopwords=False, removeNumbers=False, removeSmileys=False):
    
    # use BeautifulSoup library to remove the HTML/XML tags (e.g., <br />)
    reviewText = BeautifulSoup(rawReview).get_text()

    # Emotional symbols may affect the meaning of the review
    smileys = """:-) :) :o) :] :3 :c) :> =] 8) =) :} :^)
                :D 8-D 8D x-D xD X-D XD =-D =D =-3 =3 B^D :( :/ :-( :'( :D :P""".split()
    smiley_pattern = "|".join(map(re.escape, smileys))

    # [^] matches a single character that is not contained within the brackets
    # re.sub() replaces the pattern by the desired character/string
    
	# Check to see how we need to perform cleanup
    if removeNumbers and removeSmileys:
        reviewText = re.sub("[^a-zA-Z]", " ", reviewText)
    elif removeSmileys:
        reviewText = re.sub("[^a-zA-Z0-9]", " ", reviewText)
    elif removeNumbers:
        reviewText = re.sub("[^a-zA-Z" + smiley_pattern + "]", " ", reviewText)
    else:
        reviewText = re.sub("[^a-zA-Z0-9" + smiley_pattern + "]", " ", reviewText)

    # split in to a list of words
    words = reviewText.lower().split()

    if removeStopwords:
        # create a set of all stop words
        stops = set(stopwords.words("english"))
        # remove stop words from the list
        words = [w for w in words if w not in stops]
               
    return words