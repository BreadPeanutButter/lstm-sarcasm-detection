import pandas as pd

df = pd.read_csv('sarcasm_train.csv',engine='python')
stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 
'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 
'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 
'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 
'down', 'in', 'out', 'on', 'off', 'over', 'under', 'then', 'once', 'here', 
'there', 'when', 'where',  'how', 'any', 'both', 'each', 'few', 'more', 'most', 
'other', 'some', 'own', 'same', 'than', 'too',
 's', 't', 'can', 'will', 'don', 'now', 'd', 
 'll', 'm', 'o', 're', 've', 'y', 'ain']
df['comment'] = df['comment'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop)]))
df = df[df['comment'].notna()]
df = df[df['comment'] != '']
df.to_csv('sarcasm_train_2.csv',index = False)

