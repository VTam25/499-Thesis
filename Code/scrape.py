from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os

# Function to remove tags
def remove_tags(html):
 
    # parse html content
    soup = BeautifulSoup(html, "html.parser")
    soup = soup.find("div", {"id": "main-content"})
 
    for data in soup(['style', 'script']):
        # Remove tags
        data.decompose()
 
    # return data by retrieving the tag content
    return ' '.join(soup.stripped_strings)
 

#open raw files directory
raw_dir = "../CICS_Web_Raw_HTML"
files = os.listdir(raw_dir)

titles = np.array([])

for f in files:
    with open(raw_dir+'/'+f, encoding='utf8') as page:
        stripped = remove_tags(page)
        print("__________________NEXT DOC__________________________")
        print(f)
        print("__________________PAGE CONTENT__________________")
        print(stripped)
        new_title = f[:len(f)-8]+"PARSE.html"
        print(new_title)
        new_path = "../CICS_Web_Parsed/" + new_title
        parsed_file = open(new_path, "w", encoding='utf-8',)
        parsed_file.write(stripped)



