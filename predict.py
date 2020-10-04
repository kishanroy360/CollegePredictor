import pickle
import pandas as pd
ip = {'GRE Score': [362.5],
      'TOEFL Score': [170],
      'University Rating': [3],
      'SOP': [4.0],
      'LOR': [4.0],
      'CGPA': [7.90],
      'Research': [0]
      }
ip_df = pd.DataFrame(data=ip)
#
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict(ip_df))

print(2)