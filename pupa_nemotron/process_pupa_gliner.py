# from gliner import GLiNER
import pandas

import cld3

def is_english(text, thresh=0.70):
    r = cld3.get_language(text)
    if not r:
        return False
    return r.language == "en" and r.probability >= thresh

def determine_english_query(user_query: str):
    initial = " ".join(user_query.split(" ")[:10])
    final = " ".join(user_query.split(" ")[-10:])
    all_flags = [is_english(initial), is_english(final), is_english(user_query)]
    all_flags.sort()
    if all_flags == [False, True, True] or all_flags == [True, True, True]:
        return True
    return False
    
    

if __name__ == "__main__":
    pupa_tnb = pandas.read_csv("../pupa/PUPA_TNB.csv")
    
    # for u in list(pupa_tnb["user_query"]):
    #     print(determine_english_query(u), u)
    
    # pupa_tnb["is_english_query"] = [False] * len(pupa_tnb)
    print(len(pupa_tnb))
    pupa_tnb["is_english_query"] = pupa_tnb["user_query"].map(determine_english_query)
    pupa_tnb = pupa_tnb.loc[pupa_tnb["is_english_query"] == True]
    print(len(pupa_tnb))
    pupa_tnb.to_csv("../pupa/PUPA_TNB_ENG.csv", index=False)