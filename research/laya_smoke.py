import time, json
t=time.time()
from laya import Agent
a = Agent("convaiinnovations/laya", device="cpu")
print("load_s", round(time.time()-t,2))
q = {"topic":{"type":"choice","instructions":"What is the topic of `article`?","criteria":{"world":"world news","sports":"sports","business":"business","sci_tech":"science and technology"}}}
st = {"article":"The Lakers beat the Celtics 110-102 in overtime on Sunday."}
for i in range(3):
    t=time.time(); r=a.predict(st,q); print("ms", round((time.time()-t)*1000,1))
print(json.dumps(r["answers"]["topic"]))
