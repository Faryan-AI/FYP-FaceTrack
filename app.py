import streamlit as st
import cv2, os, av
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
import onnxruntime as ort
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import plotly.express as px

st.set_page_config(layout="wide")

# ================= CONFIG =================
KNOWN_DIR = "known_faces"
ATT_FILE = "attendance.csv"

if not os.path.exists(ATT_FILE):
    pd.DataFrame(columns=["name","time"]).to_csv(ATT_FILE,index=False)

# ================= MODELS =================
session = ort.InferenceSession("arcface.onnx")

face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# ================= MEDIAPIPE =================
mp_face = mp.solutions.face_mesh
mesh = mp_face.FaceMesh(refine_landmarks=True)

LEFT = [33,160,158,133,153,144]
RIGHT = [362,385,387,263,373,380]

# ================= FUNCTIONS =================
def detect_faces(img):
    h,w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),1.0,(300,300),(104,117,123))
    face_net.setInput(blob)
    det = face_net.forward()
    faces=[]
    for i in range(det.shape[2]):
        if det[0,0,i,2]>0.5:
            box=det[0,0,i,3:7]*np.array([w,h,w,h])
            faces.append(box.astype(int))
    return faces

def get_embedding(face):
    face=cv2.resize(face,(112,112))
    face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB).astype(np.float32)
    face=(face/255.0-0.5)/0.5
    face=np.transpose(face,(2,0,1))[np.newaxis,:]
    return session.run(None,{session.get_inputs()[0].name:face})[0].flatten()

def eye_ratio(lm,pts,w,h):
    p=[(int(lm[i].x*w),int(lm[i].y*h)) for i in pts]
    A=np.linalg.norm(np.array(p[1])-np.array(p[5]))
    B=np.linalg.norm(np.array(p[2])-np.array(p[4]))
    C=np.linalg.norm(np.array(p[0])-np.array(p[3]))
    return (A+B)/(2*C)

def is_live(img):
    h,w=img.shape[:2]
    res=mesh.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return False
    lm=res.multi_face_landmarks[0].landmark
    ear=(eye_ratio(lm,LEFT,w,h)+eye_ratio(lm,RIGHT,w,h))/2
    return ear>0.22

# ================= LOAD FACES =================
known={}
for f in os.listdir(KNOWN_DIR):
    img=cv2.imread(os.path.join(KNOWN_DIR,f))
    if img is None: continue
    faces=detect_faces(img)
    if not faces: continue
    x1,y1,x2,y2=faces[0]
    emb=get_embedding(img[y1:y2,x1:x2])
    known[os.path.splitext(f)[0]]=emb

# ================= ATTENDANCE =================
cooldown={}
def mark(name):
    now=datetime.now()
    if name in cooldown and now-cooldown[name]<timedelta(seconds=10):
        return
    cooldown[name]=now
    df=pd.read_csv(ATT_FILE)
    df=pd.concat([df,pd.DataFrame([[name,now]],columns=["name","time"])])
    df.to_csv(ATT_FILE,index=False)

# ================= STREAM =================
class Processor(VideoTransformerBase):
    def transform(self,frame):
        img=frame.to_ndarray(format="bgr24")

        if not is_live(img):
            cv2.putText(img,"SPOOF DETECTED",(20,50),0,1,(0,0,255),2)
            return img

        faces=detect_faces(img)

        for (x1,y1,x2,y2) in faces:
            face=img[y1:y2,x1:x2]
            emb=get_embedding(face)

            best="Unknown";score=0
            for n,e in known.items():
                s=cosine_similarity([emb],[e])[0][0]
                if s>score:
                    best=n;score=s

            if score>0.55:
                mark(best)
                color=(0,255,0)
            else:
                color=(0,0,255)

            cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
            cv2.putText(img,f"{best} {score:.2f}",(x1,y1-10),0,0.6,color,2)

        return img

# ================= UI =================
st.title("🎓 AI Attendance System (Professional FYP)")

tab1,tab2=st.tabs(["📷 Live System","📊 Dashboard"])

with tab1:
    webrtc_streamer(key="cam",video_transformer_factory=Processor)

with tab2:
    df=pd.read_csv(ATT_FILE)

    st.dataframe(df)

    if not df.empty:
        df["time"]=pd.to_datetime(df["time"])
        chart=df.groupby("name").size().reset_index(name="count")
        fig=px.bar(chart,x="name",y="count",title="Attendance Count")
        st.plotly_chart(fig)