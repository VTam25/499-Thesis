from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os

#open raw files directory
raw_dir = "../CICS_Web_Raw_HTML"
files = os.listdir(raw_dir)

titles = np.array([])

for f in files:
    with open(raw_dir+'/'+f) as page:
        soup = BeautifulSoup(page, "html.parser")
        print(f)
        print("__________________PAGE CONTENT__________________")
        p_tags = soup.body.find_all('p')
        for p in p_tags:
            print(p) #p.string has issues if the next characters aren't immediately strings. ex. a <a href> tag immediately after
        print("__________________NEXT DOC__________________________")



